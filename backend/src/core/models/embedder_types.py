from enum import Enum

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


class EmbedderTypes(str, Enum):
    OPENAI = "OPENAI"
    NVIDIA = "NVIDIA"
    VERTEX = "VERTEX"
    LLAMA_OLLAMA = "LLAMA-OLLAMA"


EMBEDDER_TO_CLASS = {
    "NVIDIA": NVIDIAEmbeddings,
    "OPENAI": OpenAIEmbeddings,
    "VERTEX": VertexAIEmbeddings,
    "LLAMA-OLLAMA": OllamaEmbeddings,
}
