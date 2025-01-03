from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from backend.config.settings import BaseObject, Config


class VectorStoreManager(BaseObject):
    def __init__(
            self,
            config: Config = None,
            embedder=None,
    ):
        super().__init__()
        self.config = config if config is not None else Config()
        self._embeddings = embedder
        self._retriever = None

    @property
    def retriever(self):
        return self._retriever

    def create_vector_store(self, docs: List[Document]) -> Chroma:
        vector_store = Chroma.from_documents(docs, self._embeddings)
        self._retriever = vector_store.as_retriever()
        return vector_store

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        if not self._retriever:
            raise ValueError("Vector store chưa được khởi tạo.")
        return self._retriever.get_relevant_documents(query, k=k)
