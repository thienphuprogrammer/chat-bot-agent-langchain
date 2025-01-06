from typing import Optional

from langchain_ollama import OllamaEmbeddings, ChatOllama

from backend.src.core.chains import BaseChain
from backend.src.core.models import ModelTypes
from backend.src.core.rag.relevance.fusion import FusionRelevance
from backend.src.core.rag.retrieval import PDFRetrieval
from backend.src.core.utils import VectorStoreManager
from backend.src.utils.prompt import *


class PDFQAChain(BaseChain):
    def __init__(
            self,
            vector_store_manager: VectorStoreManager,
            multi_prompt_template: str = FUSION_PROMPT,
            final_rag_prompt: str = FINAL_RAG_PROMPT,
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
        self._multi_prompt_template = self._init_prompt_template(multi_prompt_template)
        self._final_rag_prompt = self._init_prompt_template(final_rag_prompt)
        self._init_generate_chain(self._multi_prompt_template)
        self._init_retrieval_chain(FusionRelevance.reciprocal_rank_fusion)
        self._init_final_rag_chain(self._final_rag_prompt)


if __name__ == "__main__":
    # Khởi tạo các dependencies
    embedder = OllamaEmbeddings(model="llama3.2:1b")
    model = ChatOllama(model="llama3.2:1b")
    query = """
        How OmniPred work?
    """
    # Khởi tạo PDFRetriever
    pdf_retriever = PDFRetrieval(embedder=embedder, model=model)
    pdf_retriever.process_and_store_pdf(pdf_path="../../../data/pdf/OmniPred.pdf")

    # Khởi tạo QA chains
    pdf_qa_chain = PDFQAChain(vector_store_manager=pdf_retriever.vector_store_manager, base_model=model)

    # Chạy QA chains

    result = pdf_qa_chain(query)
    print(result)
