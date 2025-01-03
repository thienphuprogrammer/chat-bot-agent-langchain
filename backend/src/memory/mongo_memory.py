import logging

from backend.config.settings import Config
from langchain_community.chat_message_histories.mongodb import MongoDBChatMessageHistory

from backend.src.memory.base_memory import BaseChatbotMemory

logger = logging.getLogger(__name__)


class MongoChatbotMemory(BaseChatbotMemory):
    def __init__(self, config: Config = None, **kwargs):
        self.config = config if config is not None else Config()
        super(MongoChatbotMemory, self).__init__(
            config=config,
            chat_history_class=MongoDBChatMessageHistory,
            chat_history_kwargs={
                "connection_string": self.config.memory_connection_string,
                "session_id": self.config.session_id,
                "database_name": self.config.memory_database_name,
                "collection_name": self.config.memory_collection_name,
            }
        )
