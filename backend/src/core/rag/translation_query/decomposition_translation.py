from operator import itemgetter
from typing import List

from langchain_core.output_parsers import StrOutputParser

from backend.src.core.rag.translation_query.base_translation import BaseTranslation


class DecompositionTranslationManager(BaseTranslation):
    def __init__(
            self,
            retriever,
            embedder,
            decomposition_generate_prompt_template: str,
            decompose_prompt_template: str,
            model=None
    ):
        super().__init__(model=model, retriever=retriever, embedder=embedder)
        self._retriever = retriever
        self._base_model = model
        self._decomposition_generate_prompt_template = self._init_prompt_template(
            prompt_template=decomposition_generate_prompt_template)
        self._decompose_prompt_template = self._init_prompt_template(prompt_template=decompose_prompt_template)

        self._init_generate_chain(prompt_template=self._decomposition_generate_prompt_template)
        self._init_decompose_chain()

    @staticmethod
    def format_qa_pair(question, answer):
        """Format Q and A pair"""

        formatted_string = ""
        formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
        return formatted_string.strip()

    def _init_decompose_chain(self):
        self._decompose_chain = (
                {
                    "context": itemgetter("question") | self._retriever,
                    "question": itemgetter("question"),
                    "q_a_pairs": itemgetter("q_a_pairs")
                }
                | self._decompose_prompt_template
                | self._base_model
                | StrOutputParser()
        ).with_config(run_name="DecomposeChain")

    def _predict(self, questions: List):
        q_a_pairs, answer = "", ""
        for q in questions:
            answer = self._decompose_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
            q_a_pair = self.format_qa_pair(q, answer)
            q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
        return q_a_pairs, answer

    def __call__(self, question: str):
        questions: List = self._generate_chain.invoke({"input": question})
        return self._predict(questions)


if __name__ == "__main__":
    pass
