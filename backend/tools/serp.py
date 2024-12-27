from typing import Optional

from langchain_community.utilities import SerpAPIWrapper
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool


class CustomSearchTool(BaseTool):
    name = "Custom search"
    description = "Useful for when you need to answer questions about current or newest events, date, ..."
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
