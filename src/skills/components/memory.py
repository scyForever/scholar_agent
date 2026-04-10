from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Sequence

from src.core.models import Paper
from src.memory.manager import MemoryManager


class ResearchMemoryComponent:
    def __init__(self, memory_manager: MemoryManager | None = None) -> None:
        self.memory = memory_manager or MemoryManager()

    def remember_preference(self, user_id: str, preference: str, *, metadata: Dict[str, Any] | None = None) -> str:
        return self.memory.remember_preference(user_id, preference, metadata=metadata)

    def remember_paper(
        self,
        user_id: str,
        paper: Paper,
        summary: str,
        *,
        highlights: List[str] | None = None,
    ) -> str:
        return self.memory.remember_paper(user_id, paper, summary, highlights=highlights)

    def recall_context(self, user_id: str, query: str, *, limit: int = 8) -> List[Dict[str, Any]]:
        return [asdict(record) for record in self.memory.recall_research_context(user_id, query, limit=limit)]

    def rank_unseen_first(
        self,
        user_id: str,
        papers: Sequence[Paper],
        *,
        limit: int,
    ) -> Dict[str, Any]:
        known_keys = self.memory.seen_paper_keys(user_id)
        unseen: List[Paper] = []
        seen: List[Paper] = []
        for paper in papers:
            if self._paper_seen(paper, known_keys):
                seen.append(paper)
            else:
                unseen.append(paper)
        ranked = [*unseen, *seen][:limit]
        return {
            "papers": ranked,
            "seen_count": len(seen),
            "unseen_count": len(unseen),
            "seen_titles": [paper.title for paper in seen[:10]],
        }

    def remember_search_preferences(
        self,
        user_id: str,
        *,
        topic: str,
        time_range: str = "",
        sources: Sequence[str] | None = None,
        max_results: int | None = None,
    ) -> None:
        parts = [f"研究主题偏好：{topic}"]
        if time_range:
            parts.append(f"时间范围偏好：{time_range}")
        if sources:
            parts.append("来源偏好：" + "、".join(item for item in sources if item))
        if max_results:
            parts.append(f"检索规模偏好：{max_results} 篇")
        self.remember_preference(
            user_id,
            "\n".join(parts),
            metadata={
                "topic": topic,
                "time_range": time_range,
                "sources": list(sources or []),
                "max_results": max_results or 0,
            },
        )

    def _paper_seen(self, paper: Paper, known_keys: Iterable[str]) -> bool:
        known = set(known_keys)
        candidates = {
            (paper.paper_id or "").strip().lower(),
            (paper.title or "").strip().lower(),
            (paper.doi or "").strip().lower(),
            (paper.arxiv_id or "").strip().lower(),
            (paper.pmid or "").strip().lower(),
        }
        candidates = {item for item in candidates if item}
        return bool(candidates.intersection(known))
