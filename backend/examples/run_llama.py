import os

from langchain.prompts import PromptTemplate
import langchain.callbacks.manager as CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdoutCallback

from backend.