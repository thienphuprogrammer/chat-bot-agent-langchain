from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend import ModelTypes
from backend.common.config import BaseObject, Config
from backend.prompt import FUSION_PROMPT


class FusionObject(BaseObject):
    def __init__(
            self, config: Config = None,
            prompt_template: str = FUSION_PROMPT,
            model: Optional[ModelTypes] = None,
            model_kwargs: Optional[dict] = None,
            partial_variable: dict = None

    ):
        super().__init__()
        self.config = config if config is not None else Config()
        self.prompt_template = prompt_template

    def _init_prompt_template(self, prompt_template: str = None, partial_variables=None):
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompt_template)
        self._prompt = prompt.partial(**partial_variables)

    def _init_fusion(self, prompt_template):
        self.fusion = (
                prompt_template
                | self._prompt
                | StrOutputParser()
                | (lambda x: x.split("\n"))
        ).with_config(run_name="FusionResponse")
