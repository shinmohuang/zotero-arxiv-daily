from dataclasses import dataclass
from typing import Any, Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')
MAX_TOKENS_RANGE_PATTERN = re.compile(
    r"valid range of max_tokens is \[(\d+),\s*(\d+)\]",
    flags=re.IGNORECASE,
)
DEFAULT_SAFE_MAX_TOKENS = 8192


def _copy_generation_kwargs(llm_params: dict) -> dict[str, Any]:
    generation_kwargs = llm_params.get('generation_kwargs', {})
    if generation_kwargs is None:
        return {}
    return {key: value for key, value in generation_kwargs.items()}


def _coerce_max_tokens(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            return int(value)
    return None


def _prepare_generation_kwargs(llm_params: dict, paper_url: str) -> dict[str, Any]:
    generation_kwargs = _copy_generation_kwargs(llm_params)
    max_tokens = _coerce_max_tokens(generation_kwargs.get("max_tokens"))
    if max_tokens is not None:
        if max_tokens > DEFAULT_SAFE_MAX_TOKENS:
            logger.debug(
                f"Configured llm.generation_kwargs.max_tokens={max_tokens} exceeds the conservative limit "
                f"{DEFAULT_SAFE_MAX_TOKENS}. Clamping it for {paper_url}."
            )
            generation_kwargs["max_tokens"] = DEFAULT_SAFE_MAX_TOKENS
        else:
            generation_kwargs["max_tokens"] = max_tokens
    return generation_kwargs


def _infer_retry_max_tokens(error: Exception, generation_kwargs: dict[str, Any]) -> int | None:
    error_text = str(error)
    if "max_tokens" not in error_text.lower():
        return None

    match = MAX_TOKENS_RANGE_PATTERN.search(error_text)
    if match is not None:
        upper_bound = int(match.group(2))
        current = _coerce_max_tokens(generation_kwargs.get("max_tokens"))
        if current is None:
            return upper_bound
        if current > upper_bound:
            return upper_bound
        return None

    current = _coerce_max_tokens(generation_kwargs.get("max_tokens"))
    if current is None or current <= 1:
        return None
    return max(1, current // 2)


def _create_chat_completion(
    openai_client: OpenAI,
    messages: list[dict[str, str]],
    llm_params: dict,
    paper_url: str,
) -> Any:
    generation_kwargs = _prepare_generation_kwargs(llm_params, paper_url)
    try:
        return openai_client.chat.completions.create(
            messages=messages,
            **generation_kwargs,
        )
    except Exception as e:
        retry_max_tokens = _infer_retry_max_tokens(e, generation_kwargs)
        if retry_max_tokens is None:
            raise
        logger.warning(
            f"Retrying LLM request for {paper_url} with max_tokens={retry_max_tokens} after API rejection."
        )
        generation_kwargs["max_tokens"] = retry_max_tokens
        return openai_client.chat.completions.create(
            messages=messages,
            **generation_kwargs,
        )

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    framework_figure: Optional[bytes] = None
    framework_figure_cid: Optional[str] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        response = _create_chat_completion(
            openai_client,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            llm_params=llm_params,
            paper_url=self.url,
        )
        tldr = response.choices[0].message.content
        return tldr
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = _create_chat_completion(
                openai_client,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                llm_params=llm_params,
                paper_url=self.url,
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
