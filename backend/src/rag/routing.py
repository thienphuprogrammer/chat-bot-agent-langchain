from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from backend.config.settings import BaseObject, Config

SYSTEM_ROUTING_PROMPT = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )


class Routing(BaseObject):
    def __init__(
            self,
            config: Config = None,
            model=None,
            system_prompt_template: str = SYSTEM_ROUTING_PROMPT,
    ):
        super().__init__()
        self._base_model = model
        self.config = config if config is not None else Config()
        self._init_routing()
        self._structured_model = model.with_structured_output(RouteQuery)
        self._init_prompt_template(system_prompt_template=system_prompt_template)

    def _init_prompt_template(self, system_prompt_template: str = None) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_template),
                ("human", "{question}"),
            ]
        )
        self._prompt = prompt

    def _init_routing(self):
        self._routing = (
                self._prompt
                | self._base_model
        ).with_config(run_name="RoutingResponse")
