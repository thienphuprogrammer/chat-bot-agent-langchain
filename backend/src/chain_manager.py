from backend.src.common import BaseObject
from backend.src.core.utils.prompt import PromptUtils
from .common.constants import PERSONAL_CHAT_PROMPT_REACT
from .common.objects import Message
from .core.chains.cutom_chain import CustomChain


class ChainManager(BaseObject):
    def __init__(
            self,
            base_model=None,
            retriever=None,
            embedder=None,
            prompt_react_template: str = PERSONAL_CHAT_PROMPT_REACT,
            partial_variables: dict = None,
    ):
        super().__init__(base_model=base_model)
        self._prompt = PromptUtils().init_prompt_template_hub(template_path=prompt_react_template,
                                                              partial_variables=partial_variables)
        self.retriever = retriever
        self.embedder = embedder
        self.chain = CustomChain(base_model=base_model, prompt_template=prompt_react_template,
                                 partial_variables=partial_variables)

    async def _predict(self, message: Message, conversation_id: str):
        output = self.chain(message=message.message, conversation_id=conversation_id)
        return output

    async def __call__(self, message: Message, conversation_id: str):
        output: Message = await self._predict(message=message, conversation_id=conversation_id)
        return output


if __name__ == "__main__":
    pass
