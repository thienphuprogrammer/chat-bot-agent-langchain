from operator import itemgetter

from backend.common.objects import ChatRequest
from backend.core.models import ModelTypes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from backend.src.memory import MemoryTypes
from bot import Bot

bot = Bot(
    memory=MemoryTypes.CUSTOM_MEMORY,
    model=ModelTypes.NVIDIA,
    tools=[]
)
app = FastAPI(title="Chatbot App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_routes(
    app,
    {
        "setence": itemgetter("input"),
        "conversation_id": itemgetter("conversation_id")
    } | RunnableLambda(bot.call()),
    path="/chat",
    input_type=ChatRequest
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
