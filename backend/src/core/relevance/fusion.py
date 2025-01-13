from typing import List, Any, Tuple

from langchain.load import dumps, loads

from backend.src.common import BaseObject


class FusionRelevance(BaseObject):
    def __init__(self):
        super().__init__()

    @staticmethod
    def reciprocal_rank_fusion(results: List[List[Any]], k: int = 60) -> List[Tuple[Any, float]]:
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
            and an optional parameter k used in the RRF formula """

        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # previous_score = fused_scores[doc_str]
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return reranked_results
