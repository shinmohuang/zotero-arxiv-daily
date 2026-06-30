from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, Paper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from .pdf_figure import extract_framework_figure_from_url
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(
            api_key=config.llm.api.key, base_url=config.llm.api.base_url)

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id,
                            'user', self.config.zotero.api_key)

        # 1) collections 用于构建路径
        collections = zot.everything(zot.collections())
        collections = {c['key']: c for c in collections}

        # 2) 拉取 items：不要用 itemType='a || b || c'
        #    先尽量排除附件/笔记（减少数据量），再本地过滤
        from pyzotero.errors import HTTPError
        import time

        def fetch_with_retry(max_tries=5, base_sleep=2.0):
            last_err = None
            for i in range(max_tries):
                try:
                    # 排除 attachment / note，通常能大幅减少 items 数量与耗时
                    return zot.everything(zot.items(itemType='-attachment'))
                except HTTPError as e:
                    last_err = e
                    msg = str(e)
                    # 只对 504/502/503 这类临时性错误重试
                    if any(code in msg for code in ["Code: 504", "Code: 502", "Code: 503"]):
                        sleep_s = base_sleep * (2 ** i) + random.random()
                        logger.warning(
                            f"Zotero API temporary error ({msg.splitlines()[0]}). Retry {i+1}/{max_tries} after {sleep_s:.1f}s")
                        time.sleep(sleep_s)
                        continue
                    raise
            raise last_err

        raw = fetch_with_retry()

        # 3) 本地过滤你关心的三类
        allowed = {"conferencePaper", "journalArticle", "preprint"}
        corpus = [
            c for c in raw
            if c.get("data", {}).get("itemType") in allowed
            and c.get("data", {}).get("abstractNote", "").strip() != ""
        ]

        def get_collection_path(col_key: str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']

        for c in corpus:
            paths = [get_collection_path(col)
                     for col in c['data'].get('collections', [])]
            c['paths'] = paths

        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [
            CorpusPaper(
                title=c['data']['title'],
                abstract=c['data']['abstractNote'],
                added_date=datetime.strptime(
                    c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
                paths=c.get('paths', [])
            )
            for c in corpus
        ]

    def filter_corpus(self, corpus: list[CorpusPaper]) -> list[CorpusPaper]:
        include_path = self.config.zotero.include_path
        if not include_path:
            return corpus
        # include_path 可写单个 glob 字符串，也可写一组模式（命中任意一个即收录）。
        patterns = [include_path] if isinstance(include_path, str) else list(include_path)
        new_corpus = []
        logger.info(
            f"Selecting zotero papers matching include_path: {patterns}")
        for c in corpus:
            if any(glob_match(p, pattern) for p in c.paths for pattern in patterns):
                new_corpus.append(c)
        samples = random.sample(new_corpus, min(5, len(new_corpus)))
        samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths)
                            for c in samples])
        logger.info(
            f"Selected {len(new_corpus)} zotero papers:\n{samples}\n...")
        return new_corpus

    def generate_tldr_and_affiliations(self, papers: list[Paper]) -> None:
        # TLDR 与 affiliations 都是 I/O-bound 的 LLM 调用，逐篇串行是主要耗时来源
        # （N 篇 × 2 次调用）。用线程池并发，OpenAI 客户端可安全跨线程共享。
        def enrich(paper: Paper) -> None:
            paper.generate_tldr(self.openai_client, self.config.llm)
            paper.generate_affiliations(self.openai_client, self.config.llm)

        max_workers = self.config.executor.max_workers
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(tqdm(pool.map(enrich, papers), total=len(papers)))

    def attach_framework_figures(self, papers: list[Paper]) -> None:
        figure_config = getattr(self.config.executor, "figure", None)
        if figure_config is None or not figure_config.enabled:
            return

        logger.info("Extracting framework figures from PDFs...")

        def extract(item: tuple[int, Paper]) -> None:
            index, paper = item
            if paper.pdf_url is None:
                return
            try:
                figure = extract_framework_figure_from_url(
                    paper.pdf_url,
                    max_pages=figure_config.max_pages,
                    zoom=figure_config.zoom,
                    min_width=figure_config.min_width,
                    min_height=figure_config.min_height,
                    caption_margin=figure_config.caption_margin,
                )
            except Exception as e:
                logger.warning(f"Failed to extract framework figure of {paper.url}: {e}")
                return

            if figure is None:
                return
            paper.framework_figure = figure
            paper.framework_figure_cid = f"framework-figure-{index}"

        # 每篇都要下载 PDF + 解析，下载是 I/O-bound，并发能显著缩短总时长。
        max_workers = self.config.executor.max_workers
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(tqdm(pool.map(extract, enumerate(papers)), total=len(papers)))

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(
                f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(
            f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:
                                              self.config.executor.max_paper_num]
            logger.info("Generating TLDR and affiliations...")
            self.generate_tldr_and_affiliations(reranked_papers)
            self.attach_framework_figures(reranked_papers)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        zotero_add = getattr(self.config, "zotero_add", None)
        if zotero_add is not None and getattr(zotero_add, "enabled", False):
            email_content = render_email(
                reranked_papers,
                add_endpoint=getattr(zotero_add, "endpoint", None),
                add_token=getattr(zotero_add, "token", None),
                add_collection=getattr(zotero_add, "collection", None),
            )
        else:
            email_content = render_email(reranked_papers)
        inline_images = [
            (paper.framework_figure_cid, paper.framework_figure)
            for paper in reranked_papers
            if paper.framework_figure_cid is not None and paper.framework_figure is not None
        ]
        send_email(self.config, email_content, inline_images=inline_images)
        logger.info("Email sent successfully")
