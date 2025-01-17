from langchain.callbacks import FinalStreamingStdOutCallbackHandler

from backend.src.common import BaseObject, Config
from backend.src.core.models import ModelTypes


class ModelLoaderKwargs(BaseObject):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = config if config is not None else Config()

    def get_model_kwargs(self, model=None):
        if model and model == ModelTypes.OPENAI:
            return self.openai_model_kwargs
        elif model and model == ModelTypes.NVIDIA:
            return self.nvidia_model_kwargs
        elif model and model == ModelTypes.LLAMA_OLLAMA:
            return self.llama_ollama_model_kwargs
        else:
            return self.default_model_kwargs

    @property
    def llama_ollama_model_kwargs(self):
        return {
            "model": "llama3.2:1b",
            "temperature": 0
        }

    @property
    def default_model_kwargs(self):
        return {
            "max_output_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40
        }

    @property
    def nvidia_model_kwargs(self):
        return {
            "model_name": "meta/llama-3.1-405b-instruct",
            "temperature": 0
        }

    @property
    def openai_model_kwargs(self):
        return {
            "temperature": 0.2,
            "model_name": "gpt-3.5-turbo"
        }

    @property
    def streaming_model_kwargs(self):
        return {
            **self.default_model_kwargs,
            "streaming": True,
            "stop": ["\nObservation"],
            "callbacks": [FinalStreamingStdOutCallbackHandler()]  # Use only with agent
        }
