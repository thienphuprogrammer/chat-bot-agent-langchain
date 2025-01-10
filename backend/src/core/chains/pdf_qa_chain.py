from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_ollama import OllamaEmbeddings, ChatOllama

from backend.src.common.constants import PERSONAL_CHAT_PROMPT_REACT
from backend.src.core.chains import BaseChain
from backend.src.core.indexing.multi_presentation_indexing import MultiPresentationIndexing
from backend.src.core.processor.pdf_processor import PDFProcessor
from backend.src.core.relevance.fusion import FusionRelevance
from backend.src.core.retrieval import PDFRetrieval
from backend.src.core.utils.prompt import *

load_dotenv()


class PDFQAChain(BaseChain):
    def __init__(
            self,
            pdf_retriever: PDFRetrieval,
            base_model,
            embeddings,
            partial_variables: dict = None,
            prompt_react_template: str = PERSONAL_CHAT_PROMPT_REACT,
            multi_prompt_template: str = FUSION_PROMPT,
            final_rag_prompt: str = FINAL_RAG_PROMPT,
    ):
        super().__init__(
            base_model=base_model
        )
        self._embedder = embeddings
        self._pdf_retriever = pdf_retriever
        self._vector_store_manager = self._pdf_retriever.vector_store_manager
        self._retriever = self._vector_store_manager.get_retriever()
        self._multi_prompt_template = PromptUtils().init_prompt_template(multi_prompt_template)
        self._final_rag_prompt = PromptUtils().init_prompt_template(final_rag_prompt)
        self._pdf_retriever = PDFRetrieval(embedder=self._embedder, model=self._base_model)
        if partial_variables is None:
            partial_variables = {}
        self._react_prompt = PromptUtils().init_prompt_template_hub(template_path=prompt_react_template,
                                                                    partial_variables=partial_variables)

        self._init_generate_chain(self._multi_prompt_template)
        self._init_retrieval_chain(FusionRelevance.reciprocal_rank_fusion)
        self._init_final_rag_chain(self._final_rag_prompt)

    def _predict(self, message: str, conversation_id: str = ""):
        try:
            output = self.final_chain.invoke({"question": message})
            return output
        finally:
            wait_for_all_tracers()

    def __call__(self, message: str, conversation_id: str = ""):
        output = self._predict(message, conversation_id)
        return output


if __name__ == "__main__":
    # Khởi tạo các dependencies
    embedder = OllamaEmbeddings(model="llama3.2:1b")
    model = ChatOllama(model="llama3.2:1b")
    prompt_template = """
    You are an agent specializing in content summarization. Based on the provided context, provide the title of the document.

    Context:
    ----------
    {context}
    ----------
    Keep your answer as concise as possible, limited to one or two sentences. Just answer what the title is, nothing else
    This ensures clarity, brevity, and professionalism in the response.
    """
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(prompt_template)

    init_prompt_template = [
        "The title of the document",
    ]

    init_prompt = [Document(page_content=doc) for doc in init_prompt_template]

    # Khởi tạo PDFRetriever
    model_summary = MultiPresentationIndexing(model=model)
    pdf_processor = PDFProcessor()

    _, docs_summary = pdf_processor.process_pdf(pdf_path="../../../data/pdf/OmniPred.pdf", summary_model=model_summary)
    pdf_summary_retriever = PDFRetrieval(embedder=embedder, model=model)
    pdf_summary_retriever.store(docs_summary)


    def retrieve_and_process_docs(init_prompt_template):
        # Retrieve documents for each query in init_prompt_template
        result = [
            j
            for i in init_prompt_template
            for j in pdf_summary_retriever.retriever(search_kwargs={"k": 4}).invoke(i)
        ]
        # Join the page_content of all documents into a single string
        return '\n'.join([doc.page_content for doc in result])


    # Step 2: Create a Runnable for the retrieval and processing logic
    retrieve_chain = RunnableLambda(retrieve_and_process_docs)

    # Step 3: Define the final chain
    chain = retrieve_chain | prompt | model
    print(chain.invoke(init_prompt_template).content)

    # print(docs_summary)
    # print(pdf_summary_retriever.retriever().invoke("Main ideas of the document"))
    # chain = (
    #         {
    #             "context": pdf_summary_retriever.retriever().map()
    #                        | (lambda x: "\n".join([doc.page_content for doc in x]))
    #         }
    #         | prompt
    #         | model
    # )
    # print(chain.invoke(init_prompt))

    # docs = pdf_processor.process_pdf(pdf_path="../../../data/pdf/OmniPred.pdf")
    # pdf_retriever = PDFRetrieval(embedder=embedder, model=model)
    # pdf_retriever.store(docs)

    # Khởi tạo QA chains
    # pdf_qa_chain = PDFQAChain(pdf_retriever=pdf_retriever, base_model=model)

    # Chạy QA chains

    # result = pdf_qa_chain(query)
    # print(result)
