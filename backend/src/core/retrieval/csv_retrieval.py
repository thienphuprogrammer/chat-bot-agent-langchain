from backend.src.common import BaseObject
from backend.src.core.processor.csv_processor import CSVProcessor
from backend.src.core.utils.vectorstore import VectorStoreManager


class CSVRetriever(BaseObject):
    def __init__(
            self,
            model=None,
            embedder=None,
    ):
        super().__init__()
        self._base_model = model
        self._embeddings = embedder
        self._csv_processor = CSVProcessor()
        self._vector_store_manager = VectorStoreManager(embedder=self._embeddings)

    def process_and_store_csv(self, pdf_path: str, unstructured_data: bool = False):
        docs = self._csv_processor.load_csv(pdf_path)
        self._vector_store_manager.create_vector_store(docs)
