from typing import Callable

from langchain_core.prompts import ChatPromptTemplate

from backend.src.common import BaseObject


class BaseTranslation(BaseObject):
    def __init__(
            self,
    ):
        super().__init__()

    @staticmethod
    def _init_prompt_template(prompt_template: str = None) -> ChatPromptTemplate:
        pass

    def _init_generate_chain(self, prompt_template: ChatPromptTemplate, run_name: str = "TranslateResponse"):
        pass

    def _init_retrieval_chain(self, func: Callable, run_name: str = "RetrieveResponse"):
        pass

    def _init_final_rag_chain(self, prompt_template: ChatPromptTemplate):
        pass

    def _predict(self, question: str):
        pass

    def __call__(self, question: str):
        pass
