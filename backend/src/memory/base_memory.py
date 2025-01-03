from typing import Optional

from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory

from backend.src.common.baseobject import BaseObject
from backend.src.common.config import Config
from backend.src.common.objects import MessageTurn


class BaseChatbotMemory(BaseObject):
    __slots__ = ["_base_memory", "_memory"]

    def __init__(
            self,
            config: Config = None,
            chat_history_class=InMemoryChatMessageHistory,
            memory_class=ConversationBufferMemory,
            chat_history_kwargs: Optional[dict] = None,
            **kwargs
    ):
        """
            Base backend memory
            :param config: Config object
            :param chat_history_class: LangChain's chat history class
            :param memory_class: LangChain's memory class
            :param chat_history_kwargs: Memory class kwargs
            :param kwargs:
        """
        super().__init__()
        self.config = config if config is not None else Config()
        self._params = kwargs
        self.chats_history_kwargs = chat_history_kwargs or {}
        self._base_memory_class = chat_history_class
        self._memory = memory_class(**self._params)
        self._user_memory = dict()

    @property
    def params(self):
        if self._params:
            return self._params
        else:
            return {
                "ai_prefix": self.config.ai_prefix,
                "human_prefix": self.config.human_prefix,
                "memory_key": self.config.memory_key,
                "k": self.config.memory_window_size
            }

    @property
    def memory(self):
        return self._memory

    @property
    def user_memory(self):
        return self._user_memory

    def clear(self, conversation_id: str):
        if conversation_id in self.user_memory:
            memory = self.user_memory.pop(conversation_id)
            memory.clear()

    def load_history(self, conversations_id: str) -> str:
        if conversations_id not in self.user_memory:
            memory = self._base_memory_class(**self.chats_history_kwargs)
            self.memory.chat_memory = memory
            return ""

        self.memory.chat_memory = self.user_memory.get(conversations_id)
        return self._memory.load_memory_variables({})["history"]

    def add_message(self, message_turn: MessageTurn):
        pass
