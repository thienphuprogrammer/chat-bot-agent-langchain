import os

from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

from backend.common.config import Config
from backend.gradio_ui import BaseGradioUI

if __name__ == "__main__":
    GGML_MODEL_PATH = os.environ["GGML_MODEL_PATH"]
    config = Config()
    callback_name = CallbackManager([StreamingStdOutCallbackHandler()])

    paritial_variables = {"personality": BOT_}