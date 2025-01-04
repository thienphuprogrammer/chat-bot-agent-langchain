from operator import itemgetter
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.langchain import wait_for_all_tracers

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
        self._generate_chain = None
        self._retrieval_chain = None
        self._final_rag_chain = None

    @staticmethod
    def _init_prompt_template(prompt_template: str = None) -> ChatPromptTemplate:
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompt_template)
        return prompt

    def _init_generate_chain(self, prompt_template: ChatPromptTemplate, run_name: str = "TranslateResponse"):
        self._generate_chain = (
                prompt_template
                | self._base_model
                | StrOutputParser()
                | (lambda x: x.split("\n"))  # Split by newlines
                | (lambda x: [q for q in x if q])
        ).with_config(run_name=run_name)

    def _init_retrieval_chain(self, func: Callable, run_name: str = "RetrieveResponse"):
        self._retrieval_chain = (
                self._generate_chain
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

    def _predict(self, question: str):
        if not self._final_rag_chain:
            raise ValueError("Final RAG chain is not initialized. Please initialize the chain first.")
        if not self._retrieval_chain:
            raise ValueError("Retrieval chain is not initialized. Please initialize the chain first.")
        if not self._generate_chain:
            raise ValueError("Generate queries chain is not initialized. Please initialize the chain first.")

        try:
            output = self._final_rag_chain.invoke({"question": question})
            return output
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise e
        finally:
            wait_for_all_tracers()

    def __call__(self, question: str):
        return self._predict(question)
