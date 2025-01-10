from backend.src.core.utils.vectorstore import VectorStoreManager


class BaseRetrieval:
    def __init__(
            self,
            model,
            embedder,
            vector_store_manager: VectorStoreManager
    ):
        super().__init__()
        self._base_model = model
        self._embeddings = embedder
        self._vector_store_manager = vector_store_manager

    @property
    def vector_store_manager(self):
        return self._vector_store_manager

    @property
    def retriever(self):
        return self._vector_store_manager.get_retriever

    @property
    def vectorstore(self):
        return self._vector_store_manager.vector_store
