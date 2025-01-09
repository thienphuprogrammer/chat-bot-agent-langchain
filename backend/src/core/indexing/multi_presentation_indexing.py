import uuid
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.utils import Output

from backend.src.common import BaseObject


class MultiPresentationIndexing(BaseObject):
    def __init__(
            self,
            model,
    ):
        super().__init__()
        self._base_model = model
        self._init_chain()
        self._summaries = None

    def _init_chain(
            self,
            run_name: str = "RAGIndexing"
    ) -> None:
        self.chain = (
                {
                    "doc": lambda x: x.page_content,
                }
                | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
                | self._base_model
                | StrOutputParser()
        ).with_config(run_name=run_name)

    def _init_summaries(self, docs: list[Document], max_concurrency: int = 5) -> list[Output]:
        if not self.chain:
            self._init_chain()
        self._summaries = self.chain.batch(docs, {"max_concurrency": max_concurrency})
        return self._summaries

    def get_summarize(self, docs: list[Document], max_concurrency: int = 5) -> list[Output]:
        if not self._summaries:
            self._init_summaries(docs, max_concurrency)
        return self._summaries

    def summary_docs(
            self,
            docs: List[Document],
            id_key: str,
            max_concurrency: int = 5) -> List[Document]:

        doc_ids = [str(uuid.uuid4()) for _ in docs]
        _summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(self.get_summarize(docs=docs, max_concurrency=max_concurrency))
        ]
        return _summary_docs
