from operator import itemgetter
from typing import List, Dict, Any

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableMap, RunnableLambda
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel

from backend.src.common import BaseObject, Config
from backend.src.core.processor.pdf_processor import PDFProcessor
from backend.src.core.retrieval import PDFRetrieval
from backend.src.core.utils.prompt import RELEVANCE_DOCUMENT_PROMPT, PromptUtils


class State(BaseModel):
    input: str
    conversation_id: str
    file_path: str


class CustomAgent(BaseObject):
    def __init__(self,
                 config: Config,
                 chain,
                 memory,
                 anonymizer,
                 tools,
                 model,
                 embedder,
                 relevance_prompt_template=RELEVANCE_DOCUMENT_PROMPT
                 ):
        super().__init__()
        self.config = config if config is not None else Config()
        self._base_model = model
        self._embedder = embedder
        self._relevance_prompt_template = PromptUtils().init_prompt_template(relevance_prompt_template)

        self.tools = tools
        self.chain = chain
        self.memory = memory
        self.anonymizer = anonymizer
        self.pdf_retrieve = None
        self.brain = None
        self._workflow = StateGraph(AgentState)
        self.start()

    async def _grade_documents(self, docs, question):
        print("---CHECK RELEVANCE---")
        # llm_with_tool = self._base_model.with_structured_output(DocumentGrader)
        llm_with_tool = self._base_model
        chain = self._relevance_prompt_template | llm_with_tool
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result
        return "yes" if score == "yes" else "no"

    def start(self):
        history_loader = RunnableMap(
            {
                "input": itemgetter("input"),
                "agent_scratchpad": itemgetter("intermediate_steps") | RunnableLambda(format_log_to_str),
                "chat_history": itemgetter("conversation_id") | RunnableLambda(self.memory.load_history)
            }
        ).with_config(run_name="LoadHistory")

        if self.config.enable_anonymizer:
            anonymizer_runnable = self.anonymizer.get_runnable_anonymizer().with_config(run_name="AnonymizeSentence")
            de_anonymizer = RunnableLambda(self.anonymizer.anonymizer.deanonymize).with_config(
                run_name="DeAnonymizeResponse")

            agent = (
                    history_loader
                    | anonymizer_runnable
                    | self.chain.chain.chain
                    | de_anonymizer
                    | ReActSingleInputOutputParser()
            )
        else:
            agent = history_loader | self.chain.chain.chain | ReActSingleInputOutputParser()

        self.brain = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=2,
            return_intermediate_steps=False,
            handle_parsing_errors=True
        )

    async def route_model(self, state: State) -> str:
        file_path = state.file_path
        if not file_path:
            return "agent"
        try:
            docs = PDFProcessor().process_pdf(pdf_path=file_path)
            self.pdf_retrieve = PDFRetrieval(embedder=self._embedder, model=self._base_model)
            self.pdf_retrieve.store(docs)
            return "retrieve"
        except Exception as e:
            print(f"Error processing pdf: {e}")
            return "agent"

    async def retrieve_documents(self, state: State) -> str:
        query = state.input
        try:
            retrieved_docs: List[Document] = self.pdf_retrieve.retriever().invoke(query)
            concat_doc = "\n".join([doc.page_content for doc in retrieved_docs])
            return concat_doc
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return ""

    async def build_chain(self, state: State) -> Dict[str, Any]:
        try:
            if await self.route_model(state) == "agent":
                print("---AGENT---")
                output = await self.brain.ainvoke({"input": state.input, "conversation_id": state.conversation_id})
            else:
                docs = await self.retrieve_documents(state)
                relevance = await self._grade_documents(docs, state.input)
                if relevance == "yes":
                    inputs = f"\ncontext: \n {docs} \nquestion: \n {state.input}"
                    output = self.brain.invoke({
                        "input": inputs,
                        "conversation_id": state.conversation_id})['output']
                else:
                    output = self.brain.invoke(
                        {
                            "input": state.input,
                            "conversation_id": state.conversation_id
                        }
                    )['output']
            return output
        except Exception as e:
            print(f"Error building chain: {e}")
            raise

    async def __call__(self, message: str, conversation_id: str, file_path: str = None):
        state = State(input=message, conversation_id=conversation_id, file_path=file_path)
        output = await self.build_chain(state)
        return output
