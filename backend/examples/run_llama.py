import os

from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

from backend.bot import Bot
from backend.common.config import Config
from backend.gradio_ui import BaseGradioUI
from backend.memory import MemoryTypes
from backend.models import ModelTypes
from backend.prompt import BOT_PERSONALITY, LLAMA_PROMPT

if __name__ == "__main__":
    GGML_MODEL_PATH = os.environ["GGML_MODEL_PATH"]
    config = Config()
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    partial_variables = {"personality": BOT_PERSONALITY}
    prompt_template = PromptTemplate(
        template=LLAMA_PROMPT,
        input_varriables={"history", "input"},
        partial_variables=partial_variables
    )

    bot = Bot(
        config=config,
        prompt_template=prompt_template,
        model=ModelTypes.LLAMA_CPP,
        memory=MemoryTypes.CUSTOM_MEMORY,
        model_kwargs={
            "model_path": GGML_MODEL_PATH,
            "n_ctx": 512,
            "temperature": 0.75,
            "max_tokens": 512,
            "top_p": 0.95,
            "callback_manager": callback_manager,
            "verbose": True
        }
    )

    demo = BaseGradioUI(bot=bot)
    demo.start_demo()
