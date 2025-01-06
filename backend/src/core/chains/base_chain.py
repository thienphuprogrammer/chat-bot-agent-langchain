from operator import itemgetter
from typing import Optional, Callable

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from backend.src.common import BaseObject, Config
from backend.src.core.models import ModelTypes, MODEL_TO_CLASS


class BaseChain(BaseObject):
    def __init__(
            self,
            model_kwargs: Optional[dict] = None,
            model_name: Optional[ModelTypes] = None,
            base_model=None,
            config: Config = None,
            retriever=None
    ):
        super().__init__()
        self.config = config if config is not None else Config()
        self._base_model = base_model if base_model is not None \
            else self.get_model(model_type=model_name, parameters=model_kwargs)
        self._retriever = retriever
        self.generate_chain = None
        self.retrieval_chain = None
        self.final_rag_chain = None

    @staticmethod
    def _init_prompt_template_hub(template_path: str = None, partial_variables=None) -> PromptTemplate:
        prompt: PromptTemplate = hub.pull(template_path)
        return prompt.partial(**partial_variables)

    @staticmethod
    def _init_prompt_template(prompt_template: str) -> ChatPromptTemplate:
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompt_template)
        return prompt

    def get_model(self, model_type: Optional[ModelTypes] = None, parameters: Optional[dict] = None):
        model_name = parameters.pop("model_name", None)
        if model_type is None:
            model_type = ModelTypes.LLAMA_OLLAMA
        if model_type is not None:
            if model_type not in MODEL_TO_CLASS:
                raise ValueError(
                    f"Got unknown model type: {model_type}. "
                    f"Valid types are: {MODEL_TO_CLASS.keys()}."
                )
            model_class = MODEL_TO_CLASS[model_type]
        else:
            raise ValueError(
                "Somehow both `model_type` is None, "
                "this should never happen."
            )

        if model_type in [ModelTypes.VERTEX, ModelTypes.OPENAI, ModelTypes.NVIDIA, ModelTypes.LLAMA_OLLAMA]:
            if not model_name:
                model_name = self.config.base_model_name
            return model_class(model_name=model_name, **parameters)
        return model_class(**parameters, return_message=True)

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

        self.final_rag_chain = (
                {
                    "context": self.retrieval_chain,
                    "question": itemgetter("question")
                }
                | prompt_template
                | self._base_model
                | StrOutputParser()
        ).with_config(run_name="FinalRagChain")
