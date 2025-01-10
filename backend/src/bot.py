import asyncio
import uuid
from queue import Queue
from typing import Optional, List, Union

from langchain_core.tools import Tool
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_ollama import OllamaEmbeddings

from backend.src.agents.agent_custom import CustomAgent
from backend.src.chain_manager import ChainManager
from backend.src.common import Config, BaseObject
from backend.src.common.constants import *
from backend.src.common.objects import Message, MessageTurn
from backend.src.core.models import ModelTypes, MODEL_TO_CLASS
from backend.src.core.tools.serp_tool import SerpSearchTool
from backend.src.core.utils.prompt import *
from backend.src.memory import MEM_TO_CLASS, MemoryTypes
from backend.src.utils import CacheTypes, BotAnonymizer, ChatbotCache


class Bot(BaseObject):
    def __init__(
            self,
            embedder,
            config: Config = None,
            prompt_template: str = PERSONAL_CHAT_PROMPT_REACT,
            memory: Optional[MemoryTypes] = None,
            cache: Optional[CacheTypes] = None,
            model: Optional[ModelTypes] = None,
            memory_kwargs: Optional[dict] = None,
            model_kwargs: Optional[dict] = None,
            bot_personality: str = BOT_PERSONALITY,
            tools: List[Tool] = None,
    ):
        super().__init__()
        self.config = config if config is not None else Config()
        self.tools: List[Tool] = tools or [SerpSearchTool()]
        self.embeder = embedder
        self.base_model = self.get_model(model_type=model, parameters=model_kwargs)

        self.input_queue = Queue(maxsize=6)
        self._memory = self._init_memory(memory_type=memory, parameters=memory_kwargs)

        if cache == CacheTypes.GPT_CACHE and model != ModelTypes.OPENAI:
            self.cache = None
        self._cache = ChatbotCache.create(cache_type=cache)
        self.anonymizer = BotAnonymizer(config=self.config)
        self.chain = self._init_chain(base_model=self.base_model, prompt_template=prompt_template,
                                      bot_personality=bot_personality)

        self.custom_agent = CustomAgent(config=self.config, chain=self.chain, memory=self.memory,
                                        anonymizer=self.anonymizer, tools=self.tools, model=self.base_model,
                                        embedder=self.embeder)

    def get_model(self, model_type: Optional[ModelTypes] = None, parameters: Optional[dict] = None):
        model_name = parameters.pop("model", None)
        if model_type is None:
            model_type = ModelTypes.LLAMA_OLLAMA
        if model_type is not None:
            if model_type not in MODEL_TO_CLASS:
                raise ValueError(
                    f"Got unknown model type: {model_type}. "
                    f"Valid types are: {MODEL_TO_CLASS.keys()}."
                )
            model_class = MODEL_TO_CLASS[model_type]
        else:
            raise ValueError(
                "Somehow both `model_type` is None, "
                "this should never happen."
            )

        if model_type in [ModelTypes.VERTEX, ModelTypes.OPENAI, ModelTypes.NVIDIA, ModelTypes.LLAMA_OLLAMA]:
            if not model_name:
                model_name = self.config.base_model_name
            return model_class(model=model_name, **parameters)
        return model_class(**parameters, return_message=True)

    def _init_chain(self, base_model, prompt_template, bot_personality) -> ChainManager:
        partial_variables = {
            # "bot_personality": bot_personality or BOT_PERSONALITY,
            # "user_personality": "",
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
            "tool_names": ", ".join([tool.name for tool in self.tools])
        }

        return ChainManager(
            config=self.config,
            base_model=base_model,
            prompt_react_template=prompt_template,
            partial_variables=partial_variables
        )

    def _init_memory(self, parameters: dict = None, memory_type: Optional[MemoryTypes] = None):
        parameters = parameters or {}
        if memory_type is None:
            memory_type = MemoryTypes.BASE_MEMORY
        if memory_type is not None:
            if memory_type not in MEM_TO_CLASS:
                raise ValueError(
                    f"Got unknown memory type: {memory_type}. "
                    f"Valid types are: {MEM_TO_CLASS.keys()}."
                )
            memory_class = MEM_TO_CLASS[memory_type]
        else:
            raise ValueError(
                "Somehow both `memory` is None, "
                "this should never happen."
            )
        return memory_class(config=self.config, **parameters)

    @property
    def memory(self):
        return self._memory

    def reset_history(self, conversation_id: str = None):
        self.memory.clear(conversation_id=conversation_id)

    def add_message_to_memory(
            self,
            human_message: Union[Message, str],
            ai_message: Union[Message, str],
            conversation_id: str,
    ):
        if isinstance(human_message, str):
            human_message = Message(message=human_message, role=self.config.human_prefix)
        if isinstance(ai_message, str):
            ai_message = Message(message=ai_message, role=self.config.ai_prefix)

        turn = MessageTurn(
            human_message=human_message,
            ai_message=ai_message,
            conversation_id=conversation_id
        )
        self.memory.add_message(turn)

    async def __call__(self, message: Message, conversation_id: str, file_path: str = None):
        try:
            try:
                output = await  self.custom_agent(message=message.message,
                                                  conversation_id=conversation_id, file_path=file_path)
            except ValueError as e:
                import regex as re
                response = str(e)
                response = re.findall(r".*?Could not parse LLM output: `(.*)`", response)
                if not response:
                    raise e
                output = response[0]

            output = Message(message=output, role=self.config.ai_prefix)
            return output
        finally:
            wait_for_all_tracers()

    def predict(self, sentence: str, conversation_id: str = None, file_path: str = None):
        message = Message(message=sentence, role=self.config.human_prefix)
        output = asyncio.run(self(message, conversation_id=conversation_id, file_path=file_path))
        self.add_message_to_memory(human_message=message, ai_message=output, conversation_id=conversation_id)
        return output

    def call(self, input: str):
        return self.predict(**input)


if __name__ == "__main__":
    embedding = OllamaEmbeddings(model="qwen2.5:3b")
    bot = Bot(
        model=ModelTypes.LLAMA_OLLAMA,
        model_kwargs={"model": "qwen2.5:3b"},
        embedder=embedding,
    )
    bot.predict(sentence="Hello", conversation_id=str(uuid.uuid4()),
                file_path='./../../../data/pdf/OmniPred.pdf')
