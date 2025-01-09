from typing import List

from langchain_core.documents import Document

from backend.src.core.retrieval.base_retrieval import BaseRetrieval
from backend.src.core.utils import VectorStoreManager


class PDFRetrieval(BaseRetrieval):
    def __init__(
            self,
            model,
            embedder,
            vector_store_manager: VectorStoreManager = None
    ):
        if not vector_store_manager:
            vector_store_manager = VectorStoreManager(embedder=embedder)
        super().__init__(model=model, embedder=embedder, vector_store_manager=vector_store_manager)

    def store(self, docs: List[Document]):
        self._vector_store_manager.add_documents(docs)
