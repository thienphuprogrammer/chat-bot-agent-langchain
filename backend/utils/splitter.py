from typing import Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class SplitterDocument():
    def __init__(
            self,
            chunk_size: int,
            chunk_overlap: int,
            embedder=None,
            **kwargs
    ):
        super().__init__()
        self._base_embedder = embedder
        self._text_splitter = (
            RecursiveCharacterTextSplitter()
            .from_tiktoken_encoder(chunk_size=chunk_size,
                                   chunk_overlap=chunk_overlap, **kwargs)
        )

    def _splits_text(self, docs: Iterable[Document]):
        splits = self._text_splitter.split_documents(docs)
        return splits

    def vectorstore(self):
        splits = self._splits_text(blog_docs)
        vec = Chroma.from_documents(documents=splits,
                                    embedding=self._base_embedder)
        return vec


if __name__ == '__main__':
    import bs4
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    blog_docs = loader.load()

    splitter = SplitterDocument(chunk_size=1024, chunk_overlap=128)
    vec = splitter.vectorstore()
    print(vec)
