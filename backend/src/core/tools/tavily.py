from typing import Optional

from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import BaseTool

load_dotenv()


class TavilySearchTool(BaseTool):
    name: str = "Custom search"
    description: str = "Useful for when you need to answer questions about current or newest events, date, ..."

    def __init__(self, max_results=4):
        super().__init__()
        self._search: TavilySearchResults = TavilySearchResults(max_results=max_results)

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ):
        """Use the tool"""
        return self._search.invoke(query)


if __name__ == '__main__':
    tool = TavilySearchTool()
    print(tool.run("What is the current date?"))
