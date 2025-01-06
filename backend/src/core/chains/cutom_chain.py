from langchain_core.runnables import Runnable

from backend.src.core.chains import BaseChain


class CustomChain(BaseChain):
    def __init__(
            self,
            model_name,
            prompt_template: str,
            partial_variables: dict,
    ):
        super().__init__(model_name=model_name)
        self._prompt = self._init_prompt_template_hub(
            template_path=prompt_template,
            partial_variables=partial_variables)
        self._init_chain()

    def _init_chain(self, run_name: str = "GenerateResponse"):
        self._chain: Runnable = (
                self._prompt
                | self._base_model
        ).with_config(run_name=run_name)

    @property
    def chain(self):
        return self._chain
