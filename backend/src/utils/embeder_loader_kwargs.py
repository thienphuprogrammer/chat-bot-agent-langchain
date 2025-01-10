from backend.src.common import BaseObject
from backend.src.core.models.embedder_types import EmbedderTypes


class EmbedderLoaderKwargs(BaseObject):
    def __init__(self):
        super().__init__()

    def get_embedder_kwargs(self, model=None):
        if model and model == EmbedderTypes.NVIDIA:
            return self.nvidia_embedder_kwargs
        elif model and model == EmbedderTypes.LLAMA_OLLAMA:
            return self.llama_ollama_embedder_kwargs

        return self.default_embedder_kwargs

    @property
    def llama_ollama_embedder_kwargs(self):
        return {
            "model_name": "llama3.2:1b",
        }

    @property
    def default_embedder_kwargs(self):
        return {
            "model_name": "facebook/dpr-ctx_encoder-multiset-base",
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40
        }

    @property
    def nvidia_embedder_kwargs(self):
        return {
            "model_name": "NV-Embed-QA",
        }
