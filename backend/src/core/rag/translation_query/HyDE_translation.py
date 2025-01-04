from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.src.core.rag.translation_query.base_translation import BaseTranslation
from backend.src.utils.prompt import FUSION_PROMPT


class HyDETranslationManager(BaseTranslation):
    def __init__(
            self,
            retriever,
            embedder,
            prompt_template: str = FUSION_PROMPT,
            model=None
    ):
        super().__init__(model=model, retriever=retriever, embedder=embedder)
        self._retriever = retriever
        self._base_model = model
        self._prompt = self._init_prompt_template(prompt_template=prompt_template)
        self._init_generate_chain(prompt_template=self._prompt)
        self._init_retrieval_chain()
        self._init_final_rag_chain(prompt_template=self._prompt)

    def _init_generate_chain(self, prompt_template: ChatPromptTemplate, run_name: str = "TranslateResponse"):
        self._generate_chain = (
                prompt_template
                | self._base_model
                | StrOutputParser()
        ).with_config(run_name=run_name)

    def _init_retrieval_chain(self, func: Callable, run_name: str = "RetrieveResponse"):
        self._retrieval_chain = (
                self._generate_chain
                | self._retriever
        ).with_config(run_name=run_name)


if __name__ == "__main__":
    pass
