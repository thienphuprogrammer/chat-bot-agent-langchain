import hashlib
from enum import Enum
from typing import Optional

from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from langchain.globals import set_llm_cache
from langchain_community.cache import GPTCache
from langchain_community.cache import InMemoryCache

from backend.common.config import BaseObject

CACHE_TYPE = {
    "in_memory": InMemoryCache,
    "GPTCache": GPTCache,
}


class CacheTypes(Enum):
    IN_MEMORY = "in_memory"
    GPT_CACHE = "GPTCache"


def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}")


class ChatbotCache(BaseObject):
    @classmethod
    def create(cls, cache_type: Optional[CacheTypes] = None):
        param = {}
        if cache_type is None:
            cache_type = CacheTypes.IN_MEMORY
        cache = CACHE_TYPE[cache_type]
        if cache_type == "GPTCache":
            param = {"init_func": init_gptcache}
        set_llm_cache(cache(**param))
        return cls()
