import uuid

from langchain.retrievers import MultiVectorRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings

from backend.config.settings import BaseObject, Config


class Indexing(BaseObject):
    def __init__(
            self,
            config: Config = None,
            retriever: MultiVectorRetriever = None,
            model=None,
    ):
        super().__init__()
        self.config = config if config is not None else Config()
        self._retriever = retriever
        self._base_model = model
        self.chain = None
        self._init_chain()

    def _init_chain(self) -> None:
        self._chain = (
                {
                    "doc": lambda x: x.page_content,
                }
                | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
                | self._base_model
                | StrOutputParser()
        ).with_config(run_name="RAGIndexing")

    def summarize(self, docs: list[Document], max_concurrency: int = 5) -> list:
        return self._chain.batch(docs, {"max_concurrency": max_concurrency})


if __name__ == "__main__":
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()

    loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
    docs.extend(loader.load())

    indexing = Indexing()

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name="summaries",
                         embedding_function=OpenAIEmbeddings())

    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"

    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # Docs linked to summaries
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(indexing.summarize(docs))
    ]

    # Add
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    query = "Memory in agents"
    sub_docs = vectorstore.similarity_search(query, k=1)

    retrieved_docs = retriever.get_relevant_documents(query, n_results=1)
