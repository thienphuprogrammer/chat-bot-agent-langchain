from typing import Optional

from langchain import hub
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from backend.src.common import BaseObject, Config
from backend.src.core.models import ModelTypes, MODEL_TO_CLASS


class BaseChain(BaseObject):
    def __init__(
            self,
            model_kwargs: Optional[dict],
            model_name: Optional[ModelTypes] = None,
            base_model=None,
            config: Config = None,
    ):
        super().__init__()
        self.config = config if config is not None else Config()
        self._base_model = base_model if base_model is not None \
            else self.get_model(model_type=model_name, parameters=model_kwargs)

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
