from typing import List

from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.common.config import Config, BaseObject
from backend.common.objects import Question
from backend.prompt import MULTI_TURN_PROMPT


def get_unique_union(documents: list[List]):
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def _init_general_prompt_template(prompt_template: str = None, partial_variables=None):
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompt_template)
    return prompt.partial(**partial_variables)


class TranslationManager(BaseObject):
    def __init__(
            self,
            config: Config = None,
            model=None,
            embedder=None,
            general_prompt_template: str = MULTI_TURN_PROMPT,
            vectorstore=None,
            partial_variables: dict = None,
    ):
        super().__init__()
        self.config = config if config is not None else Config()
        self._base_model = model
        self._embedder = embedder
        self._prompt_perspectives = _init_general_prompt_template(prompt_template=general_prompt_template,
                                                                  partial_variables=partial_variables)
        self._prompt = _init_general_prompt_template(prompt_template=general_prompt_template,
                                                     partial_variables=partial_variables)
        self._init_generate_queries()
        self._init_retriever(vectorstore)
        self._init_retrieval_chain()

    def _init_retriever(self, vectorstore=None):
        self._retriever = vectorstore.as_retriever() if vectorstore else None

    def _init_retrieval_chain(self):
        self.retrieval_chain = (
                self._generate_queries
                | self._retriever.map()
                | get_unique_union
        ).with_config(run_name="RetrieveResponse")

    def _init_generate_queries(self):
        self._generate_queries = (
                self._prompt_perspectives
                | self._base_model
                | StrOutputParser()
                | (lambda x: x.split("\n"))
        ).with_config(run_name="TranslateResponse")

    def _predict(self, question: Question):
        try:
            output = self.retrieval_chain.invoke({"input": question.question})
            output = get_unique_union(output)
            return output
        except Exception as e:
            raise e
