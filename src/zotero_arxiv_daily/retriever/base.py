from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, RawPaperItem
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Type
from loguru import logger
class BaseRetriever(ABC):
    name: str
    def __init__(self, config:DictConfig):
        self.config = config
        self.retriever_config = getattr(config.source,self.name)

    @abstractmethod
    def _retrieve_raw_papers(self) -> list[RawPaperItem]:
        pass

    @abstractmethod
    def convert_to_paper(self, raw_paper:RawPaperItem) -> Paper | None:
        pass

    def retrieve_papers(self) -> list[Paper]:
        raw_papers = self._retrieve_raw_papers()
        papers = []
        logger.info("Processing papers...")
        # 用 submit + 逐个收结果，单篇处理失败只丢这一篇并记日志，不让异常中断整批
        # （exec_pool.map 会在第一个异常处整体抛出）。
        with ProcessPoolExecutor(max_workers=self.config.executor.max_workers) as exec_pool:
            futures = {exec_pool.submit(self.convert_to_paper, rp): rp for rp in raw_papers}
            for future in as_completed(futures):
                try:
                    paper = future.result()
                except Exception as e:
                    logger.warning(f"Failed to process a paper: {e}")
                    continue
                if paper is not None:
                    papers.append(paper)
        return papers

registered_retrievers = {}

def register_retriever(name:str):
    def decorator(cls):
        registered_retrievers[name] = cls
        cls.name = name
        return cls
    return decorator

def get_retriever_cls(name:str) -> Type[BaseRetriever]:
    if name not in registered_retrievers:
        raise ValueError(f"Retriever {name} not found")
    return registered_retrievers[name]