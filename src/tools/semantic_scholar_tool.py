from __future__ import annotations

from typing import List

from src.core.models import Paper
from src.tools.registry import ToolDefinition, ToolParameter, register_tool
from src.tools.research_search_tool import search_source


@register_tool(
    ToolDefinition(
        name="search_semantic_scholar",
        description="Search papers from Semantic Scholar.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
        ],
        tags=["search", "semantic-scholar"],
    )
)
def search_semantic_scholar(query: str, max_results: int = 10, time_range: str = "") -> List[Paper]:
    return search_source("semantic_scholar", query, max_results=max_results, time_range=time_range)
