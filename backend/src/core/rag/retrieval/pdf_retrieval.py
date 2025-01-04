from backend.src.common.baseobject import BaseObject
from backend.src.core.processor.pdf_processor import PDFProcessor
from backend.src.core.utils.vectorstore import VectorStoreManager


class PDFRetriever(BaseObject):
    def __init__(
            self,
            model,
            embedder,
    ):
        super().__init__()
        self._base_model = model
        self._embeddings = embedder
        self._pdf_processor = PDFProcessor()
        self._vector_store_manager = VectorStoreManager(embedder=self._embeddings)

    def get_docs(self, pdf_path: str, unstructured_data: bool = False):
        return self._pdf_processor.process_pdf(pdf_path=pdf_path, unstructured_data=unstructured_data)

    @property
    def vector_store_manager(self):
        return self._vector_store_manager

    @property
    def retriever(self):
        return self._vector_store_manager.retriever

    @property
    def vectorstore(self):
        return self._vector_store_manager.vector_store

    def process_and_store_pdf(self, pdf_path: str, unstructured_data: bool = False):
        docs = self._pdf_processor.process_pdf(pdf_path=pdf_path, unstructured_data=unstructured_data)
        self._vector_store_manager.create_vector_store(docs)
