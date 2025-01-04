from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from backend.src.common.config import BaseObject


class VectorStoreManager(BaseObject):
    def __init__(
            self,
            embedder,
    ):
        super().__init__()
        self._embeddings = embedder
        self._retriever = None
        self._vector_store = None

    @property
    def retriever(self):
        return self._retriever

    @property
    def vector_store(self):
        return self._vector_store

    def create_vector_store(self, docs: List[Document]):
        self._vector_store = Chroma.from_documents(docs, self._embeddings)
        self._retriever = self._vector_store.as_retriever()

    def retrieve(self, query: str, k: int = 5) -> tuple[str, List[Document]]:
        """Retrieve information related to a query."""
        if not self._vector_store:
            raise ValueError("Vector store not created yet.")
        retrieved_docs = self._vector_store.similarity_search(query, k=k)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\n" f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )

        return serialized, retrieved_docs
