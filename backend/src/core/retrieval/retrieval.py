from typing import List

from langchain_core.documents import Document

from backend.src.common.baseobject import BaseObject
from backend.src.common.config import Config
from backend.src.core.processor.pdf_processor import PDFProcessor
from backend.src.core.utils.vectorstore import VectorStoreManager


class PDFRetriever(BaseObject):
    def __init__(
            self,
            config: Config = None,
            model=None,
            embedder=None,
    ):
        super().__init__()
        self.config = config if config is not None else Config()
        self._base_model = model
        self._embeddings = embedder
        self._pdf_processor = PDFProcessor()
        self._vector_store_manager = VectorStoreManager(embedder=self._embeddings)

    def process_and_store_pdf(self, pdf_path: str, unstructured_data: bool = False):
        docs = self._pdf_processor.process_pdf(pdf_path, unstructured_data=unstructured_data)
        self._vector_store_manager.create_vector_store(docs)

    def ask_question(self, query: str, k: int = 5) -> List[Document]:
        """Trả lời câu hỏi dựa trên thông tin trong PDF."""
        return self._vector_store_manager.retrieve(query, k=k)
