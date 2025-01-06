from typing import List

from langchain.load import dumps, loads

from backend.src.common import BaseObject


class UniqueUnionRelevance(BaseObject):
    def __init__(
            self,

    ):
        super().__init__()

    @staticmethod
    def get_unique_union(documents: list[List]):
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]
