import os
from enum import Enum


class PromptTypes(str, Enum):
    BOT_PERSONALITY_PROMPT = "bot_personality_prompt"
    DECOMPOSITION_PROMPT = "decomposition_prompt"
    FUSION_PROMPT = "fusion_prompt"
    LLAMA_PROMPT = "llama_prompt"
    MULTI_TURN_PROMPT = "multi_turn_prompt"
    FINAL_RAG_PROMPT = "final_rag_prompt"
    RELEVANCE_DOCUMENT_PROMPT = "relevance_document_prompt"
    REWRITE_AGENT_PROMPT = "rewrite_agent_prompt"
    HYDE_TRANSLATION_PROMPT = "hyde_translation_prompt"


def load_prompt_from_file(types: PromptTypes) -> str:
    # config = Config()
    file_path: str = types.value + ".txt"
    directory = os.path.dirname(__file__)
    prompt_path = os.path.join(directory, './../prompt_template')
    file_path = os.path.join(prompt_path, file_path) if not file_path.startswith(directory) else file_path

    with open(file_path, "r") as file:
        return file.read()


BOT_PERSONALITY = load_prompt_from_file(PromptTypes.BOT_PERSONALITY_PROMPT)
LLAMA_PROMPT = load_prompt_from_file(PromptTypes.LLAMA_PROMPT)
FUSION_PROMPT = load_prompt_from_file(PromptTypes.FUSION_PROMPT)
DECOMPOSITION_PROMPT = load_prompt_from_file(PromptTypes.DECOMPOSITION_PROMPT)
MULTI_TURN_PROMPT = load_prompt_from_file(PromptTypes.MULTI_TURN_PROMPT)
FINAL_RAG_PROMPT = load_prompt_from_file(PromptTypes.FINAL_RAG_PROMPT)
RELEVANCE_DOCUMENT_PROMPT = load_prompt_from_file(PromptTypes.RELEVANCE_DOCUMENT_PROMPT)
REWRITE_AGENT_PROMPT = load_prompt_from_file(PromptTypes.REWRITE_AGENT_PROMPT)
HYDE_TRANSLATION_PROMPT = load_prompt_from_file(PromptTypes.HYDE_TRANSLATION_PROMPT)
