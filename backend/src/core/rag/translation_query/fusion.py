from typing import List, Tuple, Any

from langchain.load import dumps, loads

from backend.src.core.rag.translation_query.base_translation import BaseTranslation
from backend.src.utils.prompt import FUSION_PROMPT


class FusionObject(BaseTranslation):
    def __init__(
            self,
            retriever,
            embedder,
            prompt_template: str = FUSION_PROMPT,
            model=None
    ):
        super().__init__(model=model, retriever=retriever, embedder=embedder)
        self._retriever = retriever
        self._base_model = model
        self._prompt = self._init_prompt_template(prompt_template=prompt_template)
        self._init_generate_queries(prompt_template=self._prompt)
        self._init_retrieval_chain(func=self._reciprocal_rank_fusion)
        self._init_final_rag_chain(prompt_template=self._prompt)

    @staticmethod
    def _reciprocal_rank_fusion(results: List[List[Any]], k: int = 60) -> List[Tuple[Any, float]]:
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
            and an optional parameter k used in the RRF formula """

        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                previous_score = fused_scores[doc_str]
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return reranked_results

    def _predict(self, input_message: str):
        try:
            return self._retrieval_chain(input_message)
        finally:
            pass

    def __call__(self, query):
        return self._predict(query)


if __name__ == "__main__":
    pass
