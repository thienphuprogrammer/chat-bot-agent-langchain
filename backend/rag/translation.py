from operator import itemgetter
from typing import List

import bs4
from langchain.load import dumps, loads
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.common.config import Config, BaseObject
from backend.prompt import MULTI_TURN_PROMPT, FINAL_RAG_PROMPT
from backend.utils import SplitterDocument


def get_unique_union(documents: list[List]):
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def _init_general_prompt_template(prompt_template: str = None):
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompt_template)
    return prompt


class TranslationManager(BaseObject):
    def __init__(
            self,
            config: Config = None,
            chunk_size: int = None,
            chunk_overlap: int = None,
            model=None,
            embedder=None,
            retriever=None,
            general_prompt_template: str = MULTI_TURN_PROMPT,
            final_rag_prompt_template: str = FINAL_RAG_PROMPT,
    ):
        super().__init__()
        self.config = config if config is not None else Config()
        self._base_model = model
        self._embedder = embedder

        self._prompt_perspectives = _init_general_prompt_template(prompt_template=general_prompt_template)

        self._prompt = _init_general_prompt_template(prompt_template=final_rag_prompt_template)
        self._retriever = retriever
        self._init_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._init_generate_queries()
        self._init_retrieval_chain()
        self._init_final_rag_chain()

    def _init_text_splitter(self, chunk_size: int, chunk_overlap: int, **kwargs):
        _text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter()
        self._text_splitter = _text_splitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                                   **kwargs)

    def _init_generate_queries(self):
        self._generate_queries = (
                self._prompt_perspectives
                | self._base_model
                | StrOutputParser()
                | (lambda x: x.split("\n"))  # Split by newlines
                | (lambda x: [q for q in x if q])
        ).with_config(run_name="TranslateResponse")

    def _init_retrieval_chain(self):
        self._retrieval_chain = (
                self._generate_queries
                | self._retriever.map()
                | get_unique_union
        ).with_config(run_name="RetrieveResponse")

    def _init_final_rag_chain(self):
        self._final_rag_chain = (
                {"context": self._retrieval_chain,
                 "question": itemgetter("question")}
                | self._prompt
                | self._base_model
                | StrOutputParser()
        ).with_config(run_name="FinalRagChain")

    def predict(self, question):
        try:
            output = self._final_rag_chain.invoke({"question": question})
            return output
        finally:
            wait_for_all_tracers()


if __name__ == "__main__":
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    blog_docs = loader.load()
    embedder = OllamaEmbeddings(model="llama3.2:1b")
    model = OllamaLLM(model="llama3.2:1b", temperature=0)
    splitter = SplitterDocument(chunk_size=300, chunk_overlap=30)
    splits = splitter.splits(blog_docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedder)

    retriever = vectorstore.as_retriever()
    translation_manager = TranslationManager(
        model=model,
        embedder=embedder,
        retriever=retriever,
        chunk_size=300,
        chunk_overlap=30,
        general_prompt_template=MULTI_TURN_PROMPT
    )

    question = "What is task decomposition for LLM agents?"
    print(translation_manager.predict(question))
