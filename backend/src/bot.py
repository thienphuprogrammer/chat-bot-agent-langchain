import asyncio
from operator import itemgetter
from queue import Queue
from typing import Optional, List, Union

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema.runnable.base import RunnableMap
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import Tool
from langchain_core.tracers.langchain import wait_for_all_tracers

from backend.src.chain_manager import ChainManager
from backend.src.common import Config, BaseObject
from backend.src.common.constants import *
from backend.src.common.objects import Message, MessageTurn
from backend.src.core.models import ModelTypes
from backend.src.core.tools.serp_tool import SerpSearchTool
from backend.src.memory import MEM_TO_CLASS, MemoryTypes
from backend.src.utils import CacheTypes, BotAnonymizer, ChatbotCache
from backend.src.utils.loader_kwargs import ModelLoaderKwargs
from backend.src.utils.prompt import *


class Bot(BaseObject):
    def __init__(
            self,
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
        partial_variables = {
            "bot_personality": bot_personality or BOT_PERSONALITY,
            "user_personality": "",
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
            "tool_names": ", ".join([tool.name for tool in self.tools])
        }

        self.chain = ChainManager(
            config=self.config,
            model_name=model,
            prompt_template=prompt_template,
            model_kwargs=model_kwargs if model_kwargs else ModelLoaderKwargs().get_model_kwargs(model=model),
            partial_variables=partial_variables
        )

        self.input_queue = Queue(maxsize=6)
        self._memory = self.get_memory(memory_type=memory, parameters=memory_kwargs)
        if cache == CacheTypes.GPT_CACHE and model != ModelTypes.OPENAI:
            self.cache = None
        self._cache = ChatbotCache.create(cache_type=cache)
        self.anonymizer = BotAnonymizer(config=self.config)
        self.brain = None
        self.start()

    @property
    def memory(self):
        return self._memory

    def start(self):
        history_loader = RunnableMap(
            {
                "input": itemgetter("input"),
                "agent_scratchpad": itemgetter("intermediate_steps") | RunnableLambda(format_log_to_str),
                "history": itemgetter("conversation_id") | RunnableLambda(self.memory.load_history)
            }
        ).with_config(run_name="LoadHistory")

        if self.config.enable_anonymizer:
            anonymizer_runnable = self.anonymizer.get_runnable_anonymizer().with_config(run_name="AnonymizeSentence")
            de_anonymizer = RunnableLambda(self.anonymizer.anonymizer.deanonymize).with_config(
                run_name="DeAnonymizeResponse")

            agent = (
                    history_loader
                    | anonymizer_runnable
                    | self.chain.chain
                    | de_anonymizer
                    | ReActSingleInputOutputParser()
            )
        else:
            agent = history_loader | self.chain.chain | ReActSingleInputOutputParser()

        self.brain = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=2,
            return_intermediate_steps=False,
            handle_parsing_errors=True
        )

    def get_memory(self, parameters: dict = None, memory_type: Optional[MemoryTypes] = None):
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

    def reset_history(self, conversation_id: str = None):
        self.memory.clear(conversation_id=conversation_id)

    def add_message_to_memory(
            self,
            human_message: Union[Message, str],
            ai_message: Union[Message, str],
            conversation_id: str
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

    @staticmethod
    def find_tool_by_name(tools: List[Tool], tool_name: str):
        for tool in tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Tool with name {tool_name} not found")

    async def __call__(self, message: Message, conversation_id: str):
        try:
            try:
                output = self.brain.invoke({"input": message.message, "conversation_id": conversation_id})['output']
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

    def predict(self, sentence: dict, conversation_id: str = None):
        message = Message(message=sentence, role=self.config.human_prefix)
        output = asyncio.run(self(message, conversation_id=conversation_id))
        self.add_message_to_memory(human_message=message, ai_message=output, conversation_id=conversation_id)
        return output

    def call(self, input: str):
        return self.predict(**input)
