from typing import List, Tuple, Any

from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.config.settings import BaseObject, Config
from backend.core.utils import FUSION_PROMPT


class FusionObject(BaseObject):
    """
        A class to handle the fusion of ranked documents using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
            self, config: Config = None,
            prompt_template: str = FUSION_PROMPT,
            model=None
    ):
        super().__init__()
        self._base_model = model
        self.config = config if config is not None else Config()
        self._init_prompt_template(prompt_template=prompt_template)

    def _init_prompt_template(self, prompt_template: str = None) -> None:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        self._prompt = prompt

    def _init_generate_queries(self):
        self._generate_queries = (
                self._prompt
                | self._base_model
                | StrOutputParser()
                | (lambda x: x.split("\n"))
        ).with_config(run_name="FusionResponse")

    def _init_retrieval_chain_rag_fusion(self):
        self._retrieval_chain = (
                self._generate_queries
                | retriever.map()
                | reciprocal_rank_fusion

        ).with_config(run_name="FusionResponse")

    @staticmethod
    def _reciprocal_rank_fusion(self, results: List[List[Any]], k: int = 60) -> List[Tuple[Any, float]]:
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


if __name__ == "__main__":
    pass
