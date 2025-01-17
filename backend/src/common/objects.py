from typing import Optional, List

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Message(BaseModel):
    message: str = Field(description="User message")
    role: str = Field(description="Message role in conversation")


class MessageTurn(BaseModel):
    human_message: Message = Field(description="Human message")
    ai_message: Message = Field(description="AI message")
    conversation_id: str = Field(description="Conversation ID")


class ChatRequest(BaseModel):
    input: str
    conversation_id: Optional[str]


class Question(BaseModel):
    question: str = Field(description="User question")


class HypotheticalQuestions(BaseModel):
    """Generate hypothetical questions."""

    questions: List[str] = Field(..., description="List of questions")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class RagRequest(BaseModel):
    question: str = Field(
        description="The question that the user wants to have an answer for."
    )
    url: str = Field(description="The url of the docs where the answer is.")
    deep_read: Optional[str] = Field(
        description="Specifies weather all nested pages referenced from the starting URL should be read or not. The value should be yes or no.",
        default="no",
    )


class DocumentGrader(BaseModel):
    """Data model for grading document relevance."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


def messages_from_dict(message: dict) -> str:
    human_message = message["human_message"]
    ai_message = message["ai_message"]

    human_message = Message(message=human_message["message"], role=human_message["role"])
    ai_message = Message(message=ai_message["message"], role=ai_message["role"])
    return f"{human_message.role}: {human_message.message}\n{ai_message.role}: {ai_message.message}"
