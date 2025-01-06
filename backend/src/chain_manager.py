from langchain_core.tracers.langchain import wait_for_all_tracers

from .common.objects import Message
from .core.chains.base_chain import BaseChain


class ChainManager(BaseChain):
    def __init__(
            self,
            config=None,
            model_name=None,
            model_kwargs=None,
            prompt_template: str = None,
            partial_variables: dict = None,
    ):
        super().__init__(config=config, model_name=model_name, model_kwargs=model_kwargs)
        self._prompt = self._init_prompt_template_hub(template_path=prompt_template,
                                                      partial_variables=partial_variables)
        self._init_chain()

    def _init_chain(self, run_name: str = "GenerateResponse"):
        self.chain = (
                self._prompt
                | self._base_model
        ).with_config(run_name=run_name)

    async def _predict(self, message: Message, conversation_id: str):
        try:
            output = self.chain.invoke({"input": message.message, "conversation_id": conversation_id})
            output = Message(message=output, role=self.config.ai_prefix)
            return output
        finally:
            wait_for_all_tracers()

    def chain_stream(self, input: str, conversation_id: str):
        return self.chain.astream_log(
            {"input": input, "conversation_id": conversation_id},
            include_names=["StreamResponse"]
        )

    async def __call__(self, message: Message, conversation_id: str):
        output: Message = await self._predict(message=message, conversation_id=conversation_id)
        return output
