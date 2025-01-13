from typing import List, Dict, Any

from langchain import hub
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel

from backend.src.common import BaseObject, Config
from backend.src.core.processor.pdf_processor import PDFProcessor
from backend.src.core.retrieval import PDFRetrieval
from backend.src.core.utils.prompt import RELEVANCE_DOCUMENT_PROMPT, PromptUtils, REWRITE_AGENT_PROMPT


class State(BaseModel):
    input: str
    conversation_id: str
    file_path: str


class CustomManager(BaseObject):
    def __init__(self,
                 config: Config,
                 brain,
                 memory,
                 anonymizer,
                 tools,
                 model,
                 embedder,
                 relevance_prompt_template=RELEVANCE_DOCUMENT_PROMPT,
                 rewrite_prompt_template: str = REWRITE_AGENT_PROMPT,
                 ):
        super().__init__()
        self.config = config if config is not None else Config()
        self._base_model = model
        self._embedder = embedder
        self._relevance_prompt_template = PromptUtils().init_prompt_template(relevance_prompt_template)
        self._rewrite_prompt_template = PromptUtils().init_prompt_template(rewrite_prompt_template)

        self.tools = tools
        self.brain = brain
        self.memory = memory
        self.anonymizer = anonymizer
        self.pdf_retrieve = None
        self._workflow = StateGraph(AgentState)

    async def _grade_documents(self, docs, question):
        llm_with_tool = self._base_model
        chain = self._relevance_prompt_template | llm_with_tool
        scored_result = chain.invoke({"question": question, "context": docs})
        return "yes" if scored_result == "yes" else "no"

    def _rewrite(self, state):
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content
        chain = self._rewrite_prompt_template | self._base_model
        response = chain.invoke(question)
        return {"messages": [response]}

    def _generate(self, docs, question):
        prompt = hub.pull("rlm/rag-prompt")
        print("---GENERATE---")
        docs = "\n\n".join(doc.page_content for doc in docs)
        rag_chain = prompt | self._base_model | StrOutputParser()

        generation = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [generation]}

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

    async def retrieve_documents(self, query: str) -> List[Document]:
        try:
            retrieved_docs: List[Document] = self.pdf_retrieve.retriever().invoke(query)
            return retrieved_docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    async def build_chain(self, state: State) -> Dict[str, Any]:
        try:
            if await self.route_model(state) == "agent":
                print("---AGENT---")
                return await self.brain.ainvoke({"input": state.input, "conversation_id": state.conversation_id})

            docs = await self.retrieve_documents(state.input)
            rag_chain = self._generate(docs, state.input)
            relevant_docs = self._grade_documents(docs, state.input)
            if relevant_docs == "yes":
                inputs = f"question:\n{state.input}\ncontext:\n{docs}\n"
                return self.brain.invoke({"input": inputs, "conversation_id": state.conversation_id})['output']
            return self.brain.invoke({"input": state.input, "conversation_id": state.conversation_id})['output']
        except Exception as e:
            print(f"Error building chain: {e}")
            raise

    async def __call__(self, message: str, conversation_id: str, file_path: str = None):
        state = State(input=message, conversation_id=conversation_id, file_path=file_path)
        output = await self.build_chain(state)
        return output
