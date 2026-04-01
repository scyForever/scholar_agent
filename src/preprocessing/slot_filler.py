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
    PREVIOUS_SEARCH_MARKERS = (
        "根据之前查找到的资料",
        "根据之前查到的资料",
        "根据前面查到的资料",
        "基于之前查找到的资料",
        "结合之前查找到的资料",
        "根据之前搜索到的资料",
        "根据上一次检索到的资料",
        "根据刚才查找到的资料",
        "根据前面的检索结果",
        "根据之前的检索结果",
    )
    LOCAL_ONLY_MARKERS = (
        "只查本地",
        "只用本地",
        "仅查本地",
        "仅用本地",
        "只基于本地",
        "仅基于本地",
        "只根据本地",
        "仅根据本地",
        "只查rag",
        "只用rag",
        "仅查rag",
        "仅用rag",
        "只查知识库",
        "只用知识库",
        "仅查知识库",
        "仅用知识库",
    )
    NO_RETRIEVAL_MARKERS = (
        "不要查",
        "不用查",
        "无需检索",
        "不要检索",
        "不用检索",
        "别检索",
        "别查资料",
        "不要查资料",
        "不用查资料",
        "不需要查资料",
        "无需查资料",
    )

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

        context_source = self._extract_context_source(normalized)
        if context_source:
            slots["context_source"] = context_source

        rag_mode = self._extract_rag_mode(normalized)
        if rag_mode:
            slots["rag_mode"] = rag_mode

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
        cleaned = self._strip_context_control_phrases(query)
        cleaned = re.sub(r"请|帮我|想要|给我|搜索|检索|写一篇|生成|分析|解释|比较|介绍|直接", "", cleaned)
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

    def _extract_context_source(self, query: str) -> str:
        normalized = query.strip().lower()
        if any(marker in normalized for marker in self.PREVIOUS_SEARCH_MARKERS):
            return "previous_search"
        return ""

    def _extract_rag_mode(self, query: str) -> str:
        normalized = query.strip().lower()
        if any(marker in normalized for marker in self.LOCAL_ONLY_MARKERS):
            return "local_only"
        if any(marker in normalized for marker in self.NO_RETRIEVAL_MARKERS):
            return "off"
        return ""

    def _strip_context_control_phrases(self, query: str) -> str:
        cleaned = query
        for marker in self.PREVIOUS_SEARCH_MARKERS:
            cleaned = re.sub(re.escape(marker), "", cleaned, flags=re.IGNORECASE)
        for marker in self.LOCAL_ONLY_MARKERS:
            cleaned = re.sub(re.escape(marker), "", cleaned, flags=re.IGNORECASE)
        for marker in self.NO_RETRIEVAL_MARKERS:
            cleaned = re.sub(re.escape(marker), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(根据|基于|结合|参考)\s*", "", cleaned)
        cleaned = re.sub(r"\s*(资料|检索结果|搜索结果|搜索到的内容|查找到的资料)\s*", " ", cleaned)
        return " ".join(cleaned.split())
