from enum import Enum

from langchain_google_vertexai import ChatVertexAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class ModelTypes(str, Enum):
    OPENAI = "OPENAI"
    NVIDIA = "NVIDIA"
    VERTEX = "VERTEX"
    LLAMA_OLLAMA = "LLAMA-OLLAMA"


MODEL_TO_CLASS = {
    "NVIDIA": ChatNVIDIA,
    "OPENAI": ChatOpenAI,
    "VERTEX": ChatVertexAI,
    "LLAMA-OLLAMA": ChatOllama,
}
