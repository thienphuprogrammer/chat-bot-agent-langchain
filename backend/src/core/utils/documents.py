from typing import List

from langchain_core.documents import Document


class DocumentUtils:
    """Utility class for handling document operations."""

    @staticmethod
    def format_documents(docs: List[Document]) -> str:
        """Formats a list of documents into a string."""
        return "\n\n".join(doc.page_content for doc in docs)
