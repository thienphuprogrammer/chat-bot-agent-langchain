import uuid
from typing import List

from langchain.retrievers import MultiVectorRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryByteStore
from langchain_core.vectorstores import VectorStoreRetriever

from backend.src.common.config import BaseObject


class VectorStoreManager(BaseObject):
    def __init__(
            self,
            embedder,
            store: InMemoryByteStore = None,
            collection_name: str = None,
            persist_directory: str = None,
            multi_vector_retriever: bool = False,
    ):
        super().__init__()
        self._embeddings = embedder
        self._vector_store = None
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._multi_vector_retriever = multi_vector_retriever
        self._retrieve = None
        self._store = store if store else InMemoryByteStore()

    @property
    def retriever(
            self,
            search_type: str = None,
            search_kwargs: dict = None,
            id_key: str = None
    ) -> MultiVectorRetriever | VectorStoreRetriever:
        if not self._multi_vector_retriever:
            retriever = MultiVectorRetriever(
                vector_store=self._vector_store,
                byte_store=self._store,
                id_key=id_key
            )
            self._retrieve = retriever
        else:
            retriever = self._vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
            self._retrieve = retriever
        return retriever

    @property
    def vector_store(self):
        return self._vector_store

    def _init_vector_store(self):
        if not self._collection_name:
            raise ValueError("Collection name not provided.")
        if not self._persist_directory:
            raise ValueError("Persist directory not provided")

        self._vector_store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._persist_directory)

    def add_documents(
            self,
            docs: List[Document],
    ) -> None:
        if not self._vector_store:
            self._init_vector_store()
        doc_ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        if not self._multi_vector_retriever:
            self._vector_store.add_documents(documents=docs, metadata=doc_ids)
        else:
            retriever = self.retriever
            retriever.vectorstore.add_documents(documents=docs)
            retriever.docstore.mset(list(zip(doc_ids, docs)))

    def update_vector_store(
            self,
            docs: List[Document],
            _uuid: List[str]
    ):
        if not self._vector_store:
            raise ValueError("Vector store not created yet.")
        self._vector_store.update_documents(documents=docs, metadata=_uuid)

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
