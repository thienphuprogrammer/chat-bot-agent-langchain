from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

from backend.src.core.processor.base_processor import BaseProcessor
from backend.src.core.utils.splitter import SplitterDocument


class PDFProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def process_pdf(
            self,
            pdf_path: str,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            unstructured_data: bool = False,
    ) -> List[Document]:
        try:
            # Resolve the absolute path of the PDF file
            pdf_path = self._resolve_path(pdf_path)
            # Choose the appropriate loader based on the unstructured_data flag
            loader = UnstructuredLoader(file_path=pdf_path) if unstructured_data else PyMuPDFLoader(pdf_path)
            pages = loader.load()

            # Split the text into chunks
            text_splitter = SplitterDocument(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            cleaned_docs = text_splitter.splits(pages)
            return cleaned_docs

        except Exception as e:
            raise ValueError(f"Error processing PDF at '{pdf_path}': {e}")
