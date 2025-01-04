from operator import itemgetter
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.src.common import BaseObject


class BaseTranslation(BaseObject):
    def __init__(
            self,
            model,
            embedder,
            retriever
    ):
        super().__init__()
        self._base_model = model
        self._embedder = embedder
        self._retriever = retriever
        self._generate_queries = None
        self._retrieval_chain = None
        self._final_rag_chain = None

    @staticmethod
    def _init_prompt_template(prompt_template: str = None) -> ChatPromptTemplate:
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompt_template)
        return prompt

    def _init_generate_queries(self, prompt_template: ChatPromptTemplate, run_name: str = "TranslateResponse"):
        self._generate_queries = (
                prompt_template
                | self._base_model
                | StrOutputParser()
                | (lambda x: x.split("\n"))  # Split by newlines
                | (lambda x: [q for q in x if q])
        ).with_config(run_name=run_name)

    def _init_retrieval_chain(self, func: Callable, run_name: str = "RetrieveResponse"):
        self._retrieval_chain = (
                self._generate_queries
                | self._retriever.map()
                | func
        ).with_config(run_name=run_name)

    def _init_final_rag_chain(self, prompt_template: ChatPromptTemplate):
        self._final_rag_chain = (
                {
                    "context": self._retrieval_chain,
                    "question": itemgetter("question")
                }
                | prompt_template
                | self._base_model
                | StrOutputParser()
        ).with_config(run_name="FinalRagChain")

    def predict(self, question: str):
        pass
