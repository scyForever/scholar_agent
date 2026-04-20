from __future__ import annotations

from typing import List

from src.core.models import Paper
from src.tools.registry import ToolDefinition, ToolParameter, register_tool
from src.tools.research_search_tool import search_source


@register_tool(
    ToolDefinition(
        name="search_arxiv",
        description="Search papers from arXiv.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
        ],
        tags=["search", "arxiv"],
    )
)
def search_arxiv(query: str, max_results: int = 10, time_range: str = "") -> List[Paper]:
    return search_source("arxiv", query, max_results=max_results, time_range=time_range)
