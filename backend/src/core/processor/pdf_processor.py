from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.src.common.config import BaseObject


class PDFProcessor(BaseObject):
    def __init__(self):
        super().__init__()

    @staticmethod
    def remove_whitespace(docs: Document) -> Document:
        text = docs.page_content.replace("\n", "")
        docs.page_content = text
        return docs

    @staticmethod
    def process_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, unstructured_data: bool = False) -> \
            List[Document]:
        try:
            loader = PyMuPDFLoader(pdf_path) if not unstructured_data else UnstructuredFileLoader(file_path=pdf_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            texts = text_splitter.split_documents(pages)
            docs = [PDFProcessor.remove_whitespace(d) for d in texts]
            return docs
        except Exception as e:
            raise ValueError(f"Error processing PDF: {e}")
