from enum import Enum
from langchain_nvidia_ai_endpoints import ChatNVIDIA


class ModelTypes(str, Enum):
    OPENAI = "OPENAI"
    NVIDIA = "NVIDIA"
    VERTEX = "VERTEX"


MODEL_TO_CLASS = {
    # "OPENAI": ChatOpenAI,
    "NVIDIA": ChatNVIDIA,
}
