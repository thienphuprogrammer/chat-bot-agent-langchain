from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, ChatOllama

from backend.src.core.chains import BaseChain
from backend.src.core.models import ModelTypes
from backend.src.core.rag.retrieval import PDFRetrieval
from backend.src.core.utils import VectorStoreManager


class PDFQAChain(BaseChain):
    def __init__(
            self,
            vector_store_manager: VectorStoreManager,
            prompt_template: str,
            model_kwargs=None,
            config=None,
            model_name: Optional[ModelTypes] = None,
            base_model=None,
    ):
        super().__init__(
            config=config,
            model_name=model_name,
            model_kwargs=model_kwargs,
            base_model=base_model
        )
        self._vector_store_manager = vector_store_manager
        self._retriever = self._vector_store_manager.get_retriever()
        self._prompt = self._init_prompt_template(prompt_template)
        self._init_chain()

    def _init_chain(self, run_name: str = "GenerateResponse"):
        self.chain = (
                {
                    "context": self._retriever,
                    "query": RunnablePassthrough()
                }
                | self._prompt
                | self._base_model
                | StrOutputParser()
        ).with_config(run_name=run_name)

    def predict(self, input_message: str):
        output = self.chain.invoke(input_message)
        return output

    def __call__(self, query):
        return self.predict(query)


if __name__ == "__main__":
    template = """
        You are a information retrieval AI. Format the retrieved information as a table or text

        Use only the context for your answers, do not make up information

        query: {query}

        {context} 
        """

    # Khởi tạo các dependencies
    embedder = OllamaEmbeddings(model="llama3.2:1b")
    model = ChatOllama(model="llama3.2:1b")

    query = """
        What is the OmniPred?
    """
    vectorstore = VectorStoreManager(embedder=embedder)
    # Khởi tạo PDFRetriever
    pdf_retriever = PDFRetrieval(embedder=embedder, model=model, vector_store_manager=vectorstore)
    pdf_retriever.process_and_store_pdf(pdf_path="./../../data/pdf/OmniPred.pdf")
    # serialized, retrieved_docs = pdf_retriever.vector_store_manager.retrieve(query)

    # Khởi tạo QA chains
    pdf_qa_chain = PDFQAChain(vector_store_manager=pdf_retriever.vector_store_manager,
                              base_model=model,
                              prompt_template=template)

    # Chạy QA chains

    result = pdf_qa_chain.predict(query)
    print(result)
