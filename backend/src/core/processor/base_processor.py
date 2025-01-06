import os

from backend.src.common.config import BaseObject


class BaseProcessor(BaseObject):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _resolve_path(path: str) -> str:
        if not os.path.isabs(path):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(dir_path, path)
        if not os.path.exists(path):
            raise ValueError(f"File not found at '{path}'")
        return path
