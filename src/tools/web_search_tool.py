from __future__ import annotations

from typing import Dict, List

from src.tools.contracts import WebSearchToolRequest
from src.tools.harness import WebSearchToolHarness
from src.tools.registry import ToolDefinition, ToolParameter, register_tool
from src.tools.search_components import WebSnippetSearchComponent


WEB_SNIPPET_COMPONENT = WebSnippetSearchComponent()
WEB_SEARCH_TOOL_HARNESS = WebSearchToolHarness(WEB_SNIPPET_COMPONENT.search)


@register_tool(
    ToolDefinition(
        name="search_web",
        description="Search supplementary web snippets for CRAG validation.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回结果数量", False),
        ],
        tags=["search", "web"],
    )
)
def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    return WEB_SEARCH_TOOL_HARNESS.search(
        WebSearchToolRequest(
            query=query,
            max_results=max_results,
        )
    )
