import os
from enum import Enum


class PromptTypes(str, Enum):
    BOT_PERSONALITY_PROMPT = "bot_personality_prompt"
    DECOMPOSITION_PROMPT = "decomposition_prompt"
    FUSION_PROMPT = "fusion_prompt"
    LLAMA_PROMPT = "llama_prompt"
    MULTI_TURN_PROMPT = "multi_turn_prompt"
    FINAL_RAG_PROMPT = "final_rag_prompt"


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

if __name__ == "__main__":
    print(BOT_PERSONALITY)
    print(LLAMA_PROMPT)
    print(FUSION_PROMPT)
    print(DECOMPOSITION_PROMPT)
    print(MULTI_TURN_PROMPT)
    print(FINAL_RAG_PROMPT)
