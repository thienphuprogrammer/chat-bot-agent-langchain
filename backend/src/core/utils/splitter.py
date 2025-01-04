from typing import Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from backend.src.common import BaseObject


class SplitterDocument(BaseObject):
    def __init__(
            self,
            chunk_size: int,
            chunk_overlap: int,
            **kwargs
    ):
        super().__init__()
        self._init_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    @staticmethod
    def _remove_whitespace(doc: Document) -> Document:
        """Remove unnecessary whitespace from a document's content."""
        doc.page_content = doc.page_content.replace("\n", " ").strip()
        return doc

    def _init_text_splitter(self, chunk_size, chunk_overlap, **kwargs):
        _text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter()
        self._text_splitter = _text_splitter.from_tiktoken_encoder(chunk_size=chunk_size,
                                                                   chunk_overlap=chunk_overlap, **kwargs)

    def splits(self, docs: Iterable[Document]):
        chunks = self._text_splitter.split_documents(docs)
        cleaned_docs = [self._remove_whitespace(doc) for doc in chunks]
        return cleaned_docs
