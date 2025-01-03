from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import ChatOpenAI

from backend.src.common.config import BaseObject, Config
from backend.src.core.retrieval.retrieval import PDFRetriever


class PDFQAChain(BaseObject):
    def __init__(
            self,
            config: Config = None,
            retriever=None,
            model=None
    ):
        super().__init__()
        self.config = config if config is not None else Config()
        self._retriever = retriever
        self._model = model

    def create_chain(self):
        """Tạo QA chains."""
        template = """
        You are an information retrieval AI. Format the retrieved information as a table or text.
        Use only the context for your answers, do not make up information.

        Query: {query}

        Context: {context}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
                {"context": self._retriever, "query": RunnablePassthrough()}
                | prompt
                | self._model
                | StrOutputParser()
        )
        return chain


if __name__ == "__main__":
    # Khởi tạo các dependencies
    embedder = VertexAIEmbeddings(model_name="text-embedding-004")
    model = ChatOpenAI()

    # Khởi tạo PDFRetriever
    pdf_retriever = PDFRetriever(embedder=embedder)
    pdf_retriever.process_and_store_pdf(pdf_path="./../data/pdf/OmniPred_Language_Models_as_Universal_Regressors.pdf")

    # Khởi tạo QA chains
    pdf_qa_chain = PDFQAChain(retriever=pdf_retriever.ask_question, model=model)
    chain = pdf_qa_chain.create_chain()

    # Chạy QA chains
    query = """
    Find the details about Antimicrobial Resistance (AMR) containment and Infection Prevention Control (IPC) program.
    Break down the Approved Cost, both Total and Foreign Aid, Throwforward, and Estimated Expenditure.
    """
    result = chain.invoke(query)
    print(result)
