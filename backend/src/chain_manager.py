from .common.constants import PERSONAL_CHAT_PROMPT_REACT
from .common.objects import Message
from .core.chains import PDFQAChain
from .core.chains.base_chain import BaseChain
from .core.chains.cutom_chain import CustomChain
from .core.rag.retrieval import PDFRetrieval


class ChainManager(BaseChain):
    def __init__(
            self,
            config=None,
            model_name=None,
            model_kwargs=None,
            retriever=None,
            embedder=None,
            prompt_react_template: str = PERSONAL_CHAT_PROMPT_REACT,
            partial_variables: dict = None,
    ):
        super().__init__(config=config, model_name=model_name, model_kwargs=model_kwargs)
        self._prompt = self._init_prompt_template_hub(template_path=prompt_react_template,
                                                      partial_variables=partial_variables)
        self.retriever = retriever
        self.embedder = embedder
        self.chain = CustomChain(model_name=model_name, prompt_template=prompt_react_template,
                                 partial_variables=partial_variables)

        self.pdf_retriever = PDFRetrieval(embedder=self.embedder, model=self._base_model)
        self.pdf_qa_chain = PDFQAChain(pdf_retriever=self.pdf_retriever, base_model=self._base_model)

    def _init_pdf_qa_chain(self):
        self.pdf_qa_chain = PDFQAChain(pdf_retriever=self.pdf_retriever, base_model=self._base_model)

    async def _predict(self, message: Message, conversation_id: str, file_name: str = None):
        if file_name:
            self.pdf_retriever.process_and_store_pdf(pdf_path=file_name)
            output = self.pdf_qa_chain(message=message.message, conversation_id=conversation_id)
        else:
            output = self.chain(message=message.message, conversation_id=conversation_id)
        return output

    async def __call__(self, message: Message, conversation_id: str, file_name: str = None):
        output: Message = await self._predict(message=message, conversation_id=conversation_id, file_name=file_name)
        return output
