from backend.src.core.processor.pdf_processor import PDFProcessor
from backend.src.core.rag.retrieval.base_retrieval import BaseRetrieval


class PDFRetrieval(BaseRetrieval):
    def __init__(
            self,
            model,
            embedder,
    ):
        super().__init__(model=model, embedder=embedder)
        self._pdf_processor = PDFProcessor()

    def get_docs(self, pdf_path: str, unstructured_data: bool = False):
        return self._pdf_processor.process_pdf(pdf_path=pdf_path, unstructured_data=unstructured_data)

    def process_and_store_pdf(self, pdf_path: str, unstructured_data: bool = False):
        docs = self._pdf_processor.process_pdf(pdf_path=pdf_path, unstructured_data=unstructured_data)
        self._vector_store_manager.add_documents(docs)
