from langchain_core.runnables import Runnable
from langchain_core.tracers.langchain import wait_for_all_tracers

from backend.src.common.objects import Message
from backend.src.core.chains import BaseChain


class CustomChain(BaseChain):
    def __init__(
            self,
            model_name,
            prompt_template: str,
            partial_variables: dict,
    ):
        super().__init__(model_name=model_name)
        self._prompt = self._init_prompt_template_hub(template_path=prompt_template,
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

    def _predict(self, message: str, conversation_id: str):
        try:
            output = self.chain.invoke({"input": message, "conversation_id": conversation_id})
            output = Message(message=output, role=self.config.ai_prefix)
            return output
        finally:
            wait_for_all_tracers()

    def chain_stream(self, input: str, conversation_id: str):
        return self.chain.astream_log(
            {"input": input, "conversation_id": conversation_id},
            include_names=["StreamResponse"]
        )

    def __call__(self, message: str, conversation_id: str):
        output: Message = self._predict(message=message, conversation_id=conversation_id)
        return output
