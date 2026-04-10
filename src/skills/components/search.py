from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.core.models import Paper
from src.tools.contracts import LiteratureSearchToolRequest
from src.tools.research_search_tool import SEARCH_TOOL_HARNESS


class LiteratureSearchComponent:
    def __init__(self, memory_component: Any | None = None) -> None:
        self.memory_component = memory_component
        self.tool_harness = SEARCH_TOOL_HARNESS

    def search(
        self,
        query: str,
        *,
        platforms: Sequence[str] | None = None,
        max_results: int = 10,
        time_range: str = "",
        author: str = "",
        user_id: str = "",
    ) -> Dict[str, Any]:
        result = self.tool_harness.search(
            LiteratureSearchToolRequest(
                query=query,
                platforms=platforms or [],
                max_results=max_results,
                time_range=time_range,
                author=author,
            )
        )
        papers: List[Paper] = list(result.get("papers") or [])
        memory_trace: Dict[str, Any] = {}
        if user_id and self.memory_component is not None:
            ranked = self.memory_component.rank_unseen_first(user_id, papers, limit=max_results)
            papers = list(ranked["papers"])
            memory_trace = {
                "seen_count": ranked["seen_count"],
                "unseen_count": ranked["unseen_count"],
                "seen_titles": ranked["seen_titles"],
            }
            self.memory_component.remember_search_preferences(
                user_id,
                topic=query,
                time_range=time_range,
                sources=result.get("platforms") or [],
                max_results=max_results,
            )
        return {
            **result,
            "papers": papers[:max_results],
            "memory_trace": memory_trace,
        }
