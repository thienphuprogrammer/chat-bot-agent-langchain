import logging
from langchain.memory import MongoDBChatMessageHistory

from backend.common.config import Config
from backend.memory.base_memory import BaseChatbotMemory

logger = logging.getLogger(__name__)

class MongoChatbotMemory()