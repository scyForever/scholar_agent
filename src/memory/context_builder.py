from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.core.models import MemoryRecord, ShortTermMemory


@dataclass(slots=True)
class MemoryContextResult:
    text: str
    stats: Dict[str, Any] = field(default_factory=dict)


class MemoryContextBuilder:
    def __init__(
        self,
        *,
        max_chars: int = 5000,
        raw_ratio: float = 0.45,
        highlight_ratio: float = 0.25,
        summary_ratio: float = 0.15,
        long_ratio: float = 0.15,
    ) -> None:
        self.max_chars = max_chars
        self.raw_budget = int(max_chars * raw_ratio)
        self.highlight_budget = int(max_chars * highlight_ratio)
        self.summary_budget = int(max_chars * summary_ratio)
        self.long_budget = int(max_chars * long_ratio)

    def build(
        self,
        *,
        short_memory: ShortTermMemory,
        long_records: List[MemoryRecord],
        query: str = "",
    ) -> MemoryContextResult:
        sections: list[str] = []
        raw_text, raw_count = self._raw_section(short_memory.raw, self.raw_budget)
        highlight_text, highlight_count = self._highlight_section(short_memory.highlights, self.highlight_budget)
        summary_text = self._summary_section(short_memory.summary, self.summary_budget)
        long_text, long_count = self._long_section(long_records, self.long_budget)

        for text in (raw_text, highlight_text, summary_text, long_text):
            if text:
                sections.append(text)

        context = "\n\n".join(sections)
        if len(context) > self.max_chars:
            context = context[: self.max_chars - 3] + "..."

        return MemoryContextResult(
            text=context,
            stats={
                "query_chars": len(query),
                "context_chars": len(context),
                "max_chars": self.max_chars,
                "short_raw_used": raw_count,
                "short_raw_total": len(short_memory.raw),
                "short_highlights_used": highlight_count,
                "short_highlights_total": len(short_memory.highlights),
                "short_summary_used": bool(summary_text),
                "long_used": long_count,
                "long_total": len(long_records),
                "budget": {
                    "raw": self.raw_budget,
                    "highlights": self.highlight_budget,
                    "summary": self.summary_budget,
                    "long": self.long_budget,
                },
            },
        )

    def _raw_section(self, raw: List[Dict[str, str]], budget: int) -> tuple[str, int]:
        if not raw or budget <= 0:
            return "", 0
        selected: list[dict[str, str]] = []
        used = len("短期记忆-近期原文层：\n")
        for item in reversed(raw):
            line = f"{item.get('role', '')}：{item.get('content', '')}"
            cost = len(line) + 1
            if selected and used + cost > budget:
                break
            if not selected and cost > budget:
                line = self._truncate(line, max(budget - used, 80))
                cost = len(line) + 1
            selected.append({"role": str(item.get("role", "")), "content": line.split("：", 1)[-1]})
            used += cost
        selected.reverse()
        lines = [f"{item['role']}：{item['content']}" for item in selected]
        return "短期记忆-近期原文层：\n" + "\n".join(lines), len(selected)

    def _highlight_section(self, highlights: List[str], budget: int) -> tuple[str, int]:
        if not highlights or budget <= 0:
            return "", 0
        selected: list[str] = []
        used = len("短期记忆-重点提炼层：\n")
        for item in reversed(highlights):
            line = f"- {item}"
            cost = len(line) + 1
            if selected and used + cost > budget:
                break
            if not selected and cost > budget:
                line = self._truncate(line, max(budget - used, 80))
                cost = len(line) + 1
            selected.append(line)
            used += cost
        selected.reverse()
        return "短期记忆-重点提炼层：\n" + "\n".join(selected), len(selected)

    def _summary_section(self, summary: str, budget: int) -> str:
        if not summary or budget <= 0:
            return ""
        return "短期记忆-摘要层：\n" + self._truncate(summary, budget)

    def _long_section(self, records: List[MemoryRecord], budget: int) -> tuple[str, int]:
        if not records or budget <= 0:
            return "", 0
        lines: list[str] = []
        used = len("长期记忆-用户专属召回：\n")
        for record in records:
            keywords = record.metadata.get("keywords") or []
            keyword_text = "、".join(str(item) for item in keywords[:5]) if isinstance(keywords, list) else ""
            prefix = f"- [{record.memory_type.value} score={record.score:.3f}]"
            if keyword_text:
                prefix += f" 关键词：{keyword_text}"
            block = f"{prefix}\n{record.content}"
            cost = len(block) + 1
            if lines and used + cost > budget:
                break
            if not lines and cost > budget:
                block = self._truncate(block, max(budget - used, 120))
                cost = len(block) + 1
            lines.append(block)
            used += cost
        return "长期记忆-用户专属召回：\n" + "\n".join(lines), len(lines)

    def _truncate(self, text: str, limit: int) -> str:
        if limit <= 3:
            return ""
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
