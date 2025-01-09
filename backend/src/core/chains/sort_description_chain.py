from langchain_core.output_parsers import StrOutputParser

from backend.src.core.chains import BaseChain


class SortDescriptionChain(BaseChain):
    def __init__(self, base_model, prompt_template: str = None):
        super().__init__()
        self.base_model = base_model
        self.prompt_template = self._init_prompt_template(prompt_template)
        self.chain = self._init_chain()

    def _init_chain(self):
        return self.prompt_template | self.base_model | StrOutputParser()


if __name__ == "__main__":
    pass
