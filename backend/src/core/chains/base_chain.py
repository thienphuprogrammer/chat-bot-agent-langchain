from operator import itemgetter
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.src.common import BaseObject


class BaseChain(BaseObject):
    def __init__(
            self,
            base_model,
            retriever=None
    ):
        super().__init__()
        self._base_model = base_model
        self._retriever = retriever
        self.generate_chain = None
        self.retrieval_chain = None
        self.final_chain = None

    def _init_generate_chain(self, prompt_template: ChatPromptTemplate, run_name: str = "TranslateResponse"):
        if not self._base_model:
            raise ValueError("Base model is not initialized. Please initialize the model first.")

        self.generate_chain = (
                prompt_template
                | self._base_model
                | StrOutputParser()
                | (lambda x: x.split("\n"))  # Split by newlines
                | (lambda x: [q for q in x if q])
        ).with_config(run_name=run_name)

    def _init_retrieval_chain(self, func: Callable = None, run_name: str = "RetrieveResponse"):
        if not self.generate_chain:
            raise ValueError("Generate queries chain is not initialized. Please initialize the chain first.")
        if not self._retriever:
            raise ValueError("Retriever is not initialized. Please initialize the retriever first.")

        if not func:
            self.retrieval_chain = (
                    self.generate_chain
                    | self._retriever.map()
            ).with_config(run_name=run_name)
        else:
            self.retrieval_chain = (
                    self.generate_chain
                    | self._retriever.map()
                    | func
            ).with_config(run_name=run_name)

    def _init_final_rag_chain(self, prompt_template: ChatPromptTemplate):
        if not self.generate_chain:
            raise ValueError("Generate queries chain is not initialized. Please initialize the chain first.")
        if not self.retrieval_chain:
            raise ValueError("Retrieval chain is not initialized. Please initialize the chain first.")
        if not self._base_model:
            raise ValueError("Base model is not initialized. Please initialize the model first.")

        self.final_chain = (
                {
                    "context": self.retrieval_chain,
                    "question": itemgetter("question")
                }
                | prompt_template
                | self._base_model
                | StrOutputParser()
        ).with_config(run_name="FinalRagChain")
