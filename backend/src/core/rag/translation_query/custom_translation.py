from typing import List

from langchain.load import dumps, loads

from backend.src.core.rag.translation_query.base_translation import BaseTranslation
from backend.src.utils.prompt import MULTI_TURN_PROMPT, FINAL_RAG_PROMPT


class CustomTranslationManager(BaseTranslation):
    def __init__(
            self,
            model,
            embedder,
            retriever,
            general_prompt_template: str = MULTI_TURN_PROMPT,
            final_rag_prompt_template: str = FINAL_RAG_PROMPT,
    ):
        super().__init__(model=model, embedder=embedder, retriever=retriever)
        self._prompt_perspectives = self._init_prompt_template(
            prompt_template=general_prompt_template
        )
        self._prompt = self._init_prompt_template(
            prompt_template=final_rag_prompt_template
        )

        self._init_generate_chain(prompt_template=self._prompt_perspectives)
        self._init_retrieval_chain(func=self.get_unique_union)
        self._init_final_rag_chain(prompt_template=self._prompt)

    @staticmethod
    def get_unique_union(documents: list[List]):
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]


if __name__ == "__main__":
    # loader = WebBaseLoader(
    #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    #     bs_kwargs=dict(
    #         parse_only=bs4.SoupStrainer(
    #             class_=("post-content", "post-title", "post-header")
    #         )
    #     ),
    # )
    # blog_docs = loader.load()
    # embedder = OllamaEmbeddings(model="llama3.2:1b")
    # model = ChatOllama(model="llama3.2:1b", temperature=0)
    # splitter = SplitterDocument(chunk_size=300, chunk_overlap=30)
    # splits = splitter.splits(blog_docs)
    # vectorstore = Chroma.from_documents(documents=splits, embedding=embedder)
    #
    # retriever = vectorstore.as_retriever()
    # translation_manager = CustomTranslationManager(
    #     model=model,
    #     embedder=embedder,
    #     retriever=retriever,
    #     general_prompt_template=MULTI_TURN_PROMPT
    # )
    #
    # question = "What is task decomposition for LLM agents?"
    # print(translation_manager.predict(question))
    pass
