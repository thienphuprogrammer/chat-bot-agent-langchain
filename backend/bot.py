import asyncio
from queue import Queue
from typing import Optional, Dict, Union, List
from operator import itemgetter

from langchain.agents import AgentExecutor


from memory import MemoryTypes, MEM_TO_CLASS
from models import ModelTypes
from common.config import Config, BaseObject
from common.objects import Message, MessageTurn
from common.constants import *
from chain import ChainManager
from prompt import BOT_PERSONALITY
from utils import BotAnonymizer, CacheTypes, ChatbotCache
from tools import CustomSearchTool

class Bot(BaseObject):
    def __init__(
            self,
            config: Cofig
    ):
