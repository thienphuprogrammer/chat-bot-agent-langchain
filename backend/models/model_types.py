from enum import Enum

from langchain_community.llms import LlamaCpp
from langchain_google_vertexai import ChatVertexAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI


class ModelTypes(str, Enum):
    OPENAI = "OPENAI"
    NVIDIA = "NVIDIA"
    VERTEX = "VERTEX"


MODEL_TO_CLASS = {
    "NVIDIA": ChatNVIDIA,
    "OPENAI": ChatOpenAI,
    "VERTEX": ChatVertexAI,
    "LLAMA-CPP": LlamaCpp
}
