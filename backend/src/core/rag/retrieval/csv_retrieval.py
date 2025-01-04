from backend.src.core.processor.csv_processor import CSVProcessor
from backend.src.core.rag.retrieval.base_retrieval import BaseRetrieval


class CSVRetrieval(BaseRetrieval):
    def __init__(
            self,
            model,
            embedder,
    ):
        super().__init__(model=model, embedder=embedder)
        self._csv_processor = CSVProcessor()

    def process_and_store_csv(self, pdf_path: str):
        docs = self._csv_processor.load_csv(pdf_path)
        self._vector_store_manager.add_documents(docs)
