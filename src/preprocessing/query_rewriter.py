from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.core.llm import LLMManager
from src.core.structured_outputs import QueryRewriteOutput


REQUEST_PREFIXES = (
    "帮我找一下",
    "帮我找",
    "帮我搜索",
    "请帮我找",
    "请搜索",
    "搜索一下",
    "搜索",
    "查找",
    "查询",
    "找一下",
    "找",
    "写一篇关于",
    "写一篇有关",
    "写一篇",
    "写一个",
    "关于",
    "有关",
)

REQUEST_SUFFIXES = (
    "相关论文",
    "相关文章",
    "相关研究",
    "的文章",
    "的论文",
    "的研究",
    "综述文章",
    "综述论文",
)

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")

_DEFAULT_LLM_MANAGER: Optional[LLMManager] = None


def _get_default_llm_manager() -> LLMManager:
    global _DEFAULT_LLM_MANAGER
    if _DEFAULT_LLM_MANAGER is None:
        _DEFAULT_LLM_MANAGER = LLMManager()
    return _DEFAULT_LLM_MANAGER


def _normalize_spaces(text: str) -> str:
    return " ".join(
        text.replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("：", ":")
        .split()
    )


@dataclass(slots=True)
class QueryRewritePlan:
    core_topic: str
    english_query: str
    external_queries: List[str]
    local_queries: List[str]


class QueryRewriter:
    def __init__(self, llm: Optional[LLMManager] = None) -> None:
        self.llm = llm or _get_default_llm_manager()
        self._rewrite_cache: Dict[Tuple[str, str], QueryRewritePlan] = {}

    def plan(self, query: str, intent: str = "search_papers") -> QueryRewritePlan:
        return self._rewrite_plan(query, intent)

    def normalize_topic(self, topic: str) -> str:
        return self.extract_core_topic(topic)

    def extract_core_topic(self, topic: str) -> str:
        cleaned = _normalize_spaces(topic.strip(" ，。；;,."))
        for prefix in REQUEST_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        for suffix in REQUEST_SUFFIXES:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
        cleaned = re.sub(r"^(关于|有关)\s*", "", cleaned)
        cleaned = re.sub(r"\s*(方面|领域|方向)$", "", cleaned)
        cleaned = re.sub(
            r"\s*的(综述|研究现状|现状|最新进展|进展|方法对比|比较|对比|文章|论文|研究)$",
            "",
            cleaned,
        )
        cleaned = re.sub(r"\s*(综述|研究现状|现状|最新进展|进展|方法对比|比较|对比)$", "", cleaned)
        cleaned = cleaned.strip(" ，。；;,.")
        return cleaned or _normalize_spaces(topic)

    def rewrite(self, query: str, intent: str = "search_papers", target: str = "external") -> List[str]:
        plan = self._rewrite_plan(query, intent)
        if target == "local":
            return plan.local_queries
        return plan.external_queries

    def to_english_query(self, query: str, intent: str = "search_papers") -> str:
        plan = self._rewrite_plan(query, intent)
        return plan.english_query or plan.core_topic or _normalize_spaces(query)

    def _rewrite_plan(self, query: str, intent: str) -> QueryRewritePlan:
        normalized_query = _normalize_spaces(query.strip())
        cache_key = (normalized_query, intent)
        cached = self._rewrite_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._is_plain_english_query(normalized_query):
            plan = self._identity_plan(normalized_query)
            self._rewrite_cache[cache_key] = plan
            return plan

        if not self._has_real_provider():
            plan = self._identity_plan(normalized_query)
            self._rewrite_cache[cache_key] = plan
            return plan

        raw = self.llm.call_structured(
            self._rewrite_prompt(normalized_query, intent),
            QueryRewriteOutput,
            temperature=0.0,
            max_tokens=500,
            purpose="查询改写",
        )
        plan = self._parse_plan(raw.model_dump(mode="json"), normalized_query)
        self._rewrite_cache[cache_key] = plan
        return plan

    def _identity_plan(self, query: str) -> QueryRewritePlan:
        core_topic = self.extract_core_topic(query)
        external_queries = self._clean_queries([core_topic, query], limit=6)
        local_queries = self._clean_queries([core_topic, query], limit=8)
        return QueryRewritePlan(
            core_topic=core_topic,
            english_query=core_topic,
            external_queries=external_queries or [query],
            local_queries=local_queries or [query],
        )

    def _has_real_provider(self) -> bool:
        return any(name != "mock" for name in self.llm.providers)

    def _is_plain_english_query(self, query: str) -> bool:
        if CHINESE_RE.search(query):
            return False
        lowered = query.lower()
        return not any(lowered.startswith(prefix) for prefix in REQUEST_PREFIXES)

    def _rewrite_prompt(self, query: str, intent: str) -> str:
        return (
            "你是学术论文检索查询重写器。\n"
            "给定用户输入和任务意图，请将它重写成适合学术数据库检索的多个查询。\n"
            '只输出 JSON，格式如下：{"core_topic":"...","english_query":"...","external_queries":["..."],"local_queries":["..."]}\n'
            "字段要求：\n"
            "1. core_topic: 提炼后的核心研究主题，不包含礼貌语、动作词或冗余说明。\n"
            "2. english_query: 用于英文论文数据库的主检索式，必须是简洁的标准英文主题表达。\n"
            "3. external_queries: 3到6条，面向 arXiv、OpenAlex、Semantic Scholar、Web of Science。优先英文标准术语、常用缩写、全称、同义表达、review/survey/recent advances/comparison 等学术检索式。\n"
            "4. local_queries: 4到8条，面向本地RAG，允许中文、双语、英文扩展，但每条都必须是短检索式，不是解释句。\n"
            "5. 如果输入包含缩写、简称、中文术语或中英混合术语，请补足最常见的标准英文术语。\n"
            "6. 不要生成无关概念，不要编造陌生全称；不确定时保留用户原词。\n"
            "7. 不要输出除 JSON 以外的任何文字。\n"
            f"任务意图: {intent}\n"
            f"用户输入: {query}"
        )

    def _parse_plan(self, raw: Dict[str, Any], query: str) -> QueryRewritePlan:
        core_topic = self._clean_text(raw.get("core_topic")) or self.extract_core_topic(query)
        english_query = self._clean_text(raw.get("english_query")) or core_topic
        external_queries = self._clean_queries(raw.get("external_queries"), limit=6)
        local_queries = self._clean_queries(raw.get("local_queries"), limit=8)

        if english_query:
            external_queries = self._clean_queries([english_query, *external_queries], limit=6)
            local_queries = self._clean_queries([core_topic, english_query, *local_queries], limit=8)
        else:
            external_queries = self._clean_queries([core_topic, *external_queries], limit=6)
            local_queries = self._clean_queries([core_topic, *local_queries], limit=8)

        if not external_queries:
            external_queries = [english_query or core_topic or query]
        if not local_queries:
            local_queries = [core_topic or query]

        return QueryRewritePlan(
            core_topic=core_topic or query,
            english_query=english_query or core_topic or query,
            external_queries=external_queries,
            local_queries=local_queries,
        )

    def _clean_queries(self, raw_queries: Any, limit: int) -> List[str]:
        if isinstance(raw_queries, list):
            items = raw_queries
        elif raw_queries is None:
            items = []
        else:
            items = [raw_queries]

        deduped: List[str] = []
        seen = set()
        for item in items:
            text = self._clean_text(item)
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(text)
            if len(deduped) >= limit:
                break
        return deduped

    def _clean_text(self, value: Any) -> str:
        text = _normalize_spaces(str(value or "").strip(" ，。；;,."))
        if not text:
            return ""
        text = text.replace('"', "")
        return text[:160].strip()
