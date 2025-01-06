from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool

load_dotenv()


class DuckDuckGOSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
    _search = DuckDuckGoSearchRun()

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ):
        """Use the tool"""
        return self._search.run(query)


if __name__ == '__main__':
    tool = DuckDuckGOSearchTool()
    print(tool.run("Tesla stock price?"))
