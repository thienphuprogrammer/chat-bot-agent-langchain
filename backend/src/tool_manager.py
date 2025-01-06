from backend.src.common import BaseObject


class ToolsManager(BaseObject):
    def __init__(self, tools: dict):
        super().__init__()
        self._tools = tools

    def get_tool(self, tool_name: str):
        return self._tools[tool_name]
