from typing import Tuple, List

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama

from backend.src.common import BaseObject
from backend.src.core.rag.retrieval.pdf_retrieval import PDFRetrieval


class PDFRetrieveTool(BaseObject):
    name = "PDF Reader",
    description = "Useful for when you need to answer questions about the content of a PDF file."

    def __init__(self, model, embedder):
        super().__init__()
        self._pdf_retriever = PDFRetrieval(model=model, embedder=embedder)

    def run(self, query: str, pdf_path: str, unstructured_data: bool = False, k: int = 5) -> Tuple[str, List[Document]]:
        """
        Processes a PDF file, retrieves relevant information based on the query, and returns the results.

        Args:
            query (str): The query to search for in the PDF content.
            pdf_path (str): The path to the PDF file.
            unstructured_data (bool): Whether to process the PDF as unstructured data.
            k (int): The number of top results to retrieve.

        Returns:
            Tuple[str, List[Document]]: A tuple containing the serialized results and the retrieved documents.
        """
        # Process the PDF and create the vector store
        self._pdf_retriever.process_and_store_pdf(pdf_path=pdf_path, unstructured_data=unstructured_data)

        # Retrieve relevant information based on the query
        serialized, retrieved_docs = self._pdf_retriever.vector_store_manager.retrieve(query=query, k=k)

        return serialized, retrieved_docs


if __name__ == "__main__":
    embedder = OllamaEmbeddings(model="llama3.2:1b")
    model = ChatOllama(model="llama3.2:1b")
    # Initialize the tool with a model and embedder
    pdf_tool = PDFReadTool(model=model, embedder=embedder)

    # Run the tool with a query and PDF file
    query = "What is the main topic of the document?"
    pdf_path = "../../../data/pdf/OmniPred.pdf"
    serialized_results, retrieved_docs = pdf_tool.run(query=query, pdf_path=pdf_path)

    # Print the results
    print("Serialized Results:")
    print(serialized_results)

    print("\nRetrieved Documents:")
    for doc in retrieved_docs:
        print(f"Source: {doc.metadata}\nContent: {doc.page_content}\n")
