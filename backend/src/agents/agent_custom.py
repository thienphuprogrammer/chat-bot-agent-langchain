from operator import itemgetter

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda

from backend.src.common import BaseObject, Config


class CustomAgent(BaseObject):
    def __init__(self, config: Config, chain, memory, anonymizer, tools):
        super().__init__()
        self.config = config if config is not None else Config()
        self.tools = tools
        self.chain = chain
        self.memory = memory
        self.anonymizer = anonymizer
        self._brain = None
        self.start()

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

        self._brain = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=2,
            return_intermediate_steps=False,
            handle_parsing_errors=True
        )

    @property
    def brain(self):
        return self._brain

    def __call__(self, message: str, conversation_id: str = ""):
        output = self.brain.invoke({"input": message, "conversation_id": conversation_id})
        return output
