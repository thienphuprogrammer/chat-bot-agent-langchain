from backend.src.core.chains import BaseChain
from backend.src.core.utils.prompt import HYDE_TRANSLATION_PROMPT, PromptUtils


class HyDETranslation(BaseChain):
    def __init__(
            self,
            retriever,
            prompt_template: str = HYDE_TRANSLATION_PROMPT,
            model=None
    ):
        super().__init__(retriever=retriever, base_model=model)
        self._retriever = retriever
        self._base_model = model
        self._prompt = PromptUtils().init_prompt_template(prompt_template=prompt_template)
        self._init_generate_chain(prompt_template=self._prompt)
        self._init_retrieval_chain()
        self._init_final_rag_chain(prompt_template=self._prompt)
