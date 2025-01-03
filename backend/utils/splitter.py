from typing import Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from backend.common.config import BaseObject


class SplitterDocument(BaseObject):
    def __init__(
            self,
            chunk_size: int,
            chunk_overlap: int,
            **kwargs
    ):
        super().__init__()
        self._init_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    def _init_text_splitter(self, chunk_size, chunk_overlap, **kwargs):
        _text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter()
        self._text_splitter = _text_splitter.from_tiktoken_encoder(chunk_size=chunk_size,
                                                                   chunk_overlap=chunk_overlap, **kwargs)

    def splits(self, docs: Iterable[Document]):
        return self._text_splitter.split_documents(docs)
