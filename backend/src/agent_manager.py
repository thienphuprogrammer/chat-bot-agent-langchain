from backend.src.common import BaseObject


class AgentManager(BaseObject):
    def __init__(self):
        super().__init__()
        self._agents = dict()
