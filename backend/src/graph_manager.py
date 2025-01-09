from backend.src.common import BaseObject


class GraphManager(BaseObject):
    def __init__(self, config, graph, tools, base_model):
        super().__init__()
        self.config = config
        self.graph = graph
        self.tools = tools
        self.base_model = base_model

    def start(self):
        pass

    def __call__(self, message: str, conversation_id: str = ""):
        pass
