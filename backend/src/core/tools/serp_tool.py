from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import BaseTool

load_dotenv()


class SerpSearchTool(BaseTool):
    name: str = "Custom search"
    description: str = "Useful for when you need to answer questions about current or newest events, date, ..."
    _search = SerpAPIWrapper(params={
        "engine": "google",
        "gl": "us",
        "hl": "vi",
    })

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ):
        """Use the tool"""
        return self._search.run(query)


if __name__ == '__main__':
    tool = SerpSearchTool()
    print(tool.run("What is the current date?"))
