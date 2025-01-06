from typing import List

from langchain import hub
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools.simple import Tool
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState

from backend.src.common import BaseObject
from backend.src.common.objects import DocumentGrader
from backend.src.core.chains import BaseChain
from backend.src.utils.prompt import RELEVANCE_DOCUMENT_PROMPT, REWRITE_AGENT_PROMPT, MULTI_TURN_PROMPT


class AgenticRAG(BaseObject):
    def __init__(
            self,
            base_mode=None,
            embedder=None,
            tools: List[Tool] = None,
            relevance_prompt_template: str = RELEVANCE_DOCUMENT_PROMPT,
            rewrite_prompt_template: str = REWRITE_AGENT_PROMPT,
    ):
        super().__init__()
        self._base_model = base_mode
        self._embeddings = embedder
        self._tools = tools
        self._workflow = StateGraph(AgentState)
        self._graph = None
        self._initialize_prompt_template(relevance_prompt_template, rewrite_prompt_template)
        self.build_graph()

    def _initialize_prompt_template(self, relevance_prompt_template, rewrite_prompt_template):
        """Initialize the prompt template for document grading."""
        self._relevance_prompt_template = ChatPromptTemplate.from_template(template=relevance_prompt_template)
        self._rewrite_prompt_template = ChatPromptTemplate.from_template(template=rewrite_prompt_template)

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format a list of documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _grade_documents(self, state):
        print("---CHECK RELEVANCE---")
        llm_with_tool = self._base_model.with_structured_output(DocumentGrader)
        chain = self._relevance_prompt_template | llm_with_tool

        messages = state["messages"]
        last_message = messages[-1]
        question = messages[0].content
        docs = last_message.content

        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score

        return "_generate" if score == "yes" else "_rewrite"

    def _agent(self, state):
        print("---CALL AGENT---")
        messages = state["messages"]
        model_with_tool = self._base_model.bind_tools(self._tools)
        response = model_with_tool.invoke(messages)
        return {"messages": [response]}

    def _rewrite(self, state):
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content
        chain = self._rewrite_prompt_template | self._base_model
        response = chain.invoke(question)
        return {"messages": [response]}

    def _generate(self, state):
        prompt = hub.pull("rlm/rag-prompt")

        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content
        learn_docs = [Document(page_content=doc) for doc in docs.split("\n\n")]

        rag_chain = prompt | self._base_model | StrOutputParser()
        response = rag_chain.invoke({"context": learn_docs, "question": question})
        return {"messages": [response]}

    def build_graph(self):
        self._workflow.add_node("_agent", self._agent)
        _retrieve = ToolNode(self._tools)
        self._workflow.add_node("_retrieve", _retrieve)
        self._workflow.add_node("_rewrite", self._rewrite)
        self._workflow.add_node(
            "_generate", self._generate
        )
        self._workflow.add_edge(START, "_agent")
        self._workflow.add_conditional_edges(
            "_agent",
            tools_condition,
            {
                "tools": "_retrieve",
                END: END,
            },
        )

        # Edges taken after the `action` node is called.
        self._workflow.add_conditional_edges(
            "_retrieve",
            self._grade_documents,
        )
        self._workflow.add_edge("_generate", END)
        self._workflow.add_edge("_rewrite", "_agent")

        # Compile
        self._graph = self._workflow.compile()

    def _init_chain(self, run_name: str = "GenerateResponse"):
        translation = BaseChain(self._base_model)
        multi_prompt = MULTI_TURN_PROMPT
        self.chain = (
                multi_prompt
                | translation.generate_chain
                | self._base_model
        ).with_config(run_name=run_name)

    def _predict(self, state):
        if not self._graph:
            raise ValueError("Graph is not initialized. Please initialize the graph first.")
        try:
            output = self._graph.invoke(state)
            return output
        except Exception as e:
            print(e)
            return state

    def __call__(self, state):
        return self._predict(state)

# if __name__ == "__main__":
#     urls = [
#         "https://lilianweng.github.io/posts/2023-06-23-agent/",
#         "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#         "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
#     ]
#
#     docs = [WebBaseLoader(url).load() for url in urls]
#     docs_list = [item for sublist in docs for item in sublist]
#
#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=100, chunk_overlap=50
#     )
#     doc_splits = text_splitter.split_documents(docs_list)
#
#     # Add to vectorDB
#     vectorstore = Chroma.from_documents(
#         documents=doc_splits,
#         collection_name="rag-chroma",
#         embedding=OllamaEmbeddings(model="llama3.2:1b"),
#     )
#     retriever = vectorstore.as_retriever()
#
#     from langchain.tools.retriever import create_retriever_tool
#
#     retriever_tool = create_retriever_tool(
#         retriever,
#         "retrieve_blog_posts",
#         "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
#     )
#
#     tools = [retriever_tool]
#
#     llm = ChatOllama(temperature=0, model="llama3.2:1b")
#     rag = AgenticRAG(base_mode=llm, embedder=OllamaEmbeddings(model="llama3.2:1b"), tools=tools)
#
#     state = {
#         "messages": [
#             ("user", "What is the OmniPred?"),
#         ]
#     }
#
#     result = rag(state)
