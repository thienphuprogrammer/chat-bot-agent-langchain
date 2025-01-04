from backend.src.core.rag.indexing.indexing_base import BaseIndexing


class MultiRepresentationIndexing(BaseIndexing):
    def __init__(
            self,
            retriever=None,
            model=None,
    ):
        super().__init__(retriever=retriever, model=model)
        self._init_chain()
