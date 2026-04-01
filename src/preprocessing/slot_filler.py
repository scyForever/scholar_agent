from __future__ import annotations

import re
from typing import Any, Dict, List

from config.intent_config import INTENT_SPECS


class SlotFiller:
    YEAR_RANGE_PATTERN = re.compile(r"(20\d{2})\s*[-到至]\s*(20\d{2})")
    RECENT_YEARS_PATTERN = re.compile(r"(近|最近)\s*([一二两三四五六七八九十\d]+)\s*年")
    MAX_PAPERS_PATTERN = re.compile(r"(\d+)\s*(篇|papers?)")
    CHINESE_NUMBERS = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    def fill_slots_once(
        self,
        query: str,
        intent: str,
        current_slots: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        slots = dict(current_slots or {})
        slots.update(self.extract_slots(query, intent))
        required = INTENT_SPECS[intent].required_slots
        missing = [slot for slot in required if not slots.get(slot)]

        response: Dict[str, Any] = {"slots": slots, "missing": missing}
        if missing:
            response["ask"] = "请一次性补充以下信息：" + "、".join(missing)
        return response

    def extract_slots(self, query: str, intent: str) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        normalized = query.strip()

        year_range = self._extract_year_range(normalized)
        if year_range:
            slots["time_range"] = year_range

        max_papers = self._extract_max_papers(normalized)
        if max_papers:
            slots["max_papers"] = max_papers

        comparison_target = self._extract_comparison_target(normalized)
        if comparison_target:
            slots["comparison_target"] = comparison_target

        topic = self._extract_topic(normalized, intent)
        if topic:
            if intent == "analyze_paper":
                slots["paper_title"] = topic
            else:
                slots["topic"] = topic
        return slots

    def _extract_year_range(self, query: str) -> str:
        match = self.YEAR_RANGE_PATTERN.search(query)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        recent = self.RECENT_YEARS_PATTERN.search(query)
        if recent:
            years = self._parse_year_count(recent.group(2))
            return f"last_{years}_years" if years > 0 else ""
        return ""

    def _extract_max_papers(self, query: str) -> int:
        match = self.MAX_PAPERS_PATTERN.search(query)
        return int(match.group(1)) if match else 0

    def _extract_comparison_target(self, query: str) -> str:
        for token in ("和", "与", "vs", "对比", "compare with"):
            if token in query:
                parts = re.split(token, query, maxsplit=1)
                if len(parts) == 2:
                    return parts[1].strip(" ?，,。")
        return ""

    def _extract_topic(self, query: str, intent: str) -> str:
        if self.YEAR_RANGE_PATTERN.fullmatch(query.strip()) or self.RECENT_YEARS_PATTERN.fullmatch(query.strip()):
            return ""
        if self.MAX_PAPERS_PATTERN.fullmatch(query.strip()):
            return ""
        cleaned = re.sub(r"请|帮我|想要|给我|搜索|检索|写一篇|生成|分析|解释|比较|介绍", "", query)
        cleaned = self.YEAR_RANGE_PATTERN.sub("", cleaned)
        cleaned = self.RECENT_YEARS_PATTERN.sub("", cleaned)
        cleaned = re.sub(r"^(关于|有关)\s*", "", cleaned)
        cleaned = cleaned.replace("相关论文", "").replace("论文", "")
        cleaned = re.sub(r"\s*的$", "", cleaned).strip(" ：:，,。?？")
        if intent == "compare_methods":
            cleaned = re.split(r"和|与|vs|对比", cleaned)[0].strip()
        if not cleaned or re.fullmatch(r"[\d\s年月\-到至last_]+", cleaned, flags=re.IGNORECASE):
            return ""
        return cleaned

    def _parse_year_count(self, value: str) -> int:
        normalized = str(value or "").strip()
        if normalized.isdigit():
            return int(normalized)
        if normalized == "十":
            return 10
        if normalized.startswith("十") and len(normalized) == 2:
            return 10 + self.CHINESE_NUMBERS.get(normalized[1], 0)
        if normalized.endswith("十") and len(normalized) == 2:
            return self.CHINESE_NUMBERS.get(normalized[0], 0) * 10
        if "十" in normalized and len(normalized) == 3:
            return self.CHINESE_NUMBERS.get(normalized[0], 0) * 10 + self.CHINESE_NUMBERS.get(normalized[2], 0)
        return self.CHINESE_NUMBERS.get(normalized, 0)
