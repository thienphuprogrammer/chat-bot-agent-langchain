from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class Message(BaseModel):
    message: str = Field(description="User message")
    role: str = Field(description="Message role in conversation")


class MessageTurn(BaseModel):
    human_message: Message = Field(description="Human message")
    ai_message: Message = Field(description="AI message")
    conversation_id: str = Field(description="Conversation ID")


class ChatRequest(BaseModel):
    input: str
    conversation_id: Option[str]


def message_from_dict(message: dict)-> str:
    human_message = message["human_message"]
    ai_message = message["ai_message"]

    human_message = Message(message=human_message["message"], role=human_message["role"])
    ai_message = Message(message=ai_message["message"], role=ai_message["role"])
    return f"{human_message.role}: {human_message.message}\n{ai_message.role}: {ai_message.message}"
