from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence

from src.core.models import Paper
from src.tools.contracts import LiteratureSearchToolRequest, LiteratureSourceSearchRequest
from src.tools.harness import LiteratureSearchToolHarness
from src.tools.registry import ToolDefinition, ToolParameter, register_tool
from src.tools.search_components import LiteratureSearchService, SearchRequest


SEARCH_SERVICE = LiteratureSearchService()
SEARCH_TOOL_HARNESS = LiteratureSearchToolHarness(SEARCH_SERVICE)


def search_source(source_name: str, query: str, max_results: int = 10, time_range: str = "", author: str = "") -> List[Paper]:
    return SEARCH_TOOL_HARNESS.search_source(
        LiteratureSourceSearchRequest(
            source_name=source_name,
            query=query,
            max_results=max_results,
            time_range=time_range,
            author=author,
        )
    )


@register_tool(
    ToolDefinition(
        name="search_google_scholar",
        description="Search papers from Google Scholar.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
        ],
        tags=["search", "google-scholar"],
    )
)
def search_google_scholar(query: str, max_results: int = 10, time_range: str = "") -> List[Paper]:
    return search_source("google_scholar", query, max_results=max_results, time_range=time_range)


@register_tool(
    ToolDefinition(
        name="search_pubmed",
        description="Search papers from PubMed.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
        ],
        tags=["search", "pubmed"],
    )
)
def search_pubmed(query: str, max_results: int = 10, time_range: str = "") -> List[Paper]:
    return search_source("pubmed", query, max_results=max_results, time_range=time_range)


@register_tool(
    ToolDefinition(
        name="search_ieee_xplore",
        description="Search papers from IEEE Xplore.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
        ],
        tags=["search", "ieee-xplore"],
    )
)
def search_ieee_xplore(query: str, max_results: int = 10, time_range: str = "") -> List[Paper]:
    return search_source("ieee_xplore", query, max_results=max_results, time_range=time_range)


@register_tool(
    ToolDefinition(
        name="search_literature",
        description="Search papers across multiple academic platforms.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("platforms", "list", "检索平台列表", False),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
            ToolParameter("author", "str", "作者名", False),
        ],
        tags=["search", "literature"],
    )
)
def search_literature(
    query: str,
    platforms: List[str] | None = None,
    max_results: int = 10,
    time_range: str = "",
    author: str = "",
) -> Dict[str, Any]:
    result = SEARCH_TOOL_HARNESS.search(
        LiteratureSearchToolRequest(
            query=query,
            platforms=platforms or [],
            max_results=max_results,
            time_range=time_range,
            author=author,
        )
    )
    return {
        **result,
        "papers": [asdict(paper) for paper in result["papers"]],
    }


def search_platforms(
    query: str,
    *,
    platforms: Sequence[str] | None = None,
    max_results: int = 10,
    time_range: str = "",
    author: str = "",
) -> Dict[str, Any]:
    return SEARCH_TOOL_HARNESS.search(
        LiteratureSearchToolRequest(
            query=query,
            platforms=platforms or [],
            max_results=max_results,
            time_range=time_range,
            author=author,
        )
    )


__all__ = [
    "SearchRequest",
    "LiteratureSearchService",
    "SEARCH_SERVICE",
    "SEARCH_TOOL_HARNESS",
    "search_source",
    "search_platforms",
    "search_literature",
    "search_google_scholar",
    "search_pubmed",
    "search_ieee_xplore",
]
