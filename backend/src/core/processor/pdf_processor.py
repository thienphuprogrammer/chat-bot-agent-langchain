from typing import List, Tuple

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

from backend.src.core.indexing.multi_presentation_indexing import MultiPresentationIndexing
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
            summary_model: MultiPresentationIndexing = None,
            unstructured_data: bool = False,
    ) -> List[Document] | Tuple[List[Document], List[Document]]:
        try:
            # Resolve the absolute path of the PDF file
            pdf_path = self._resolve_path(pdf_path)
            # Choose the appropriate loader based on the unstructured_data flag
            loader = UnstructuredLoader(file_path=pdf_path) if unstructured_data else PyMuPDFLoader(pdf_path)
            pages = loader.load()
            summary_pages = None
            if summary_model is not None:
                summary_pages = summary_model.summary_docs(pages, id_key="id")

            # Split the text into chunks
            text_splitter = SplitterDocument(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            cleaned_docs = text_splitter.splits(pages)

            return cleaned_docs if summary_model is None else (cleaned_docs, summary_pages)
        except Exception as e:
            raise ValueError(f"Error processing PDF at '{pdf_path}': {e}")
