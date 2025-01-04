from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from backend.src.common.config import BaseObject
from backend.src.core.rag.retrieval.pdf_retrieval import PDFRetriever


class PDFQAChain(BaseObject):
    def __init__(
            self,
            retriever: PDFRetriever,
            base_model,
            prompt_template: str,
            graph_builder: StateGraph = None,
    ):
        super().__init__()
        self._retriever = retriever
        self._retrieve = self._retriever.vector_store_manager.retrieve
        self._base_model = base_model
        self.tools = ToolNode([self._retrieve])
        self._init_prompt_template(prompt_template=prompt_template)

    def _init_prompt_template(self, prompt_template: str = None):
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompt_template)
        self._prompt = prompt

    async def _predict(self, input_message: str) -> dict:
        return self._base_model(input_message)

    async def __call__(self, query):
        return await self._predict(query)


if __name__ == "__main__":
    template = """
                You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

                Question: {question} 
                Context: {context} 
                Answer:
            """

    # Khởi tạo các dependencies
    embedder = OllamaEmbeddings(model="llama3.2:1b")
    model = ChatOllama(model="llama3.2:1b")

    query = """
        What is the OmniPred?
    """
    # Khởi tạo PDFRetriever
    pdf_retriever = PDFRetriever(embedder=embedder, model=model)
    pdf_retriever.process_and_store_pdf(pdf_path="./data/pdf/OmniPred.pdf")
    # serialized, retrieved_docs = pdf_retriever.vector_store_manager.retrieve(query)

    # Khởi tạo QA chains
    pdf_qa_chain = PDFQAChain(retriever=pdf_retriever, base_model=model, prompt_template=template)

    # Chạy QA chains

    result = pdf_qa_chain(query)
    print(result)
