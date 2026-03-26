from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from api_keys import get_named_api_key
from config.settings import settings
from src.core.models import Paper
from src.preprocessing.query_rewriter import QueryRewriter
from src.tools.registry import ToolDefinition, ToolParameter, register_tool


WOS_QUERY_STOPWORDS = {
    "review",
    "survey",
    "paper",
    "papers",
    "article",
    "articles",
    "recent",
    "advances",
    "latest",
    "research",
    "background",
    "definition",
    "methodology",
    "comparison",
    "benchmark",
}

WOS_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+_.-]*")


def _matches_time_range(year: int | None, time_range: str) -> bool:
    if not time_range or year is None:
        return True
    if time_range.startswith("last_") and time_range.endswith("_years"):
        years = int(time_range.split("_")[1])
        return year >= datetime.utcnow().year - years + 1
    if "-" in time_range:
        start, end = time_range.split("-", 1)
        return int(start) <= year <= int(end)
    return True


def _time_range_clause(time_range: str) -> str:
    if not time_range:
        return ""
    if time_range.startswith("last_") and time_range.endswith("_years"):
        years = int(time_range.split("_")[1])
        end_year = datetime.utcnow().year
        start_year = end_year - years + 1
        return f"PY={start_year}-{end_year}"
    if "-" in time_range:
        start, end = time_range.split("-", 1)
        return f"PY={int(start)}-{int(end)}"
    return ""


def _topic_clause(query: str) -> str:
    english_query = QueryRewriter().to_english_query(query)
    tokens = [
        token
        for token in WOS_TOKEN_RE.findall(english_query)
        if token.lower() not in WOS_QUERY_STOPWORDS
    ]
    if tokens:
        joined = " AND ".join(f'"{token}"' for token in tokens[:6])
        return f"TS=({joined})"
    escaped = english_query.replace('"', "")
    return f'TS="{escaped}"'


def _build_query(query: str, time_range: str) -> str:
    clauses = [_topic_clause(query)]
    year_clause = _time_range_clause(time_range)
    if year_clause:
        clauses.append(year_clause)
    return " AND ".join(f"({clause})" for clause in clauses if clause)


def _parse_paper(item: Dict[str, Any], query: str) -> Optional[Paper]:
    title = str(item.get("title") or "").strip()
    if not title:
        return None

    source = item.get("source") or {}
    names = item.get("names") or {}
    links = item.get("links") or {}
    identifiers = item.get("identifiers") or {}
    keywords = item.get("keywords") or {}
    citations = item.get("citations") or []

    authors = [
        str(author.get("displayName") or author.get("wosStandard") or "").strip()
        for author in names.get("authors") or []
        if isinstance(author, dict) and (author.get("displayName") or author.get("wosStandard"))
    ]
    keyword_list = [
        str(keyword).strip()
        for keyword in keywords.get("authorKeywords") or []
        if str(keyword).strip()
    ]

    text = " ".join([title, " ".join(keyword_list)]).lower()
    query_terms = [
        token.lower()
        for token in WOS_TOKEN_RE.findall(QueryRewriter().to_english_query(query))
        if token.lower() not in WOS_QUERY_STOPWORDS
    ]
    if query_terms and not any(term in text for term in query_terms):
        return None

    citation_count = 0
    for citation in citations:
        if isinstance(citation, dict):
            citation_count = max(citation_count, int(citation.get("count") or 0))

    source_types = [str(item_type) for item_type in item.get("sourceTypes") or [] if str(item_type).strip()]
    types = [str(item_type) for item_type in item.get("types") or [] if str(item_type).strip()]
    year = source.get("publishYear")

    return Paper(
        paper_id=str(item.get("uid") or title),
        title=title,
        abstract="",
        authors=authors,
        year=int(year) if year else None,
        venue=str(source.get("sourceTitle") or ""),
        url=str(links.get("record") or ""),
        pdf_url="",
        citations=citation_count,
        source="Web of Science",
        categories=source_types or types,
        keywords=keyword_list,
        metadata={
            "doi": identifiers.get("doi", ""),
            "issn": identifiers.get("issn", ""),
            "wos_query": _build_query(query, ""),
        },
    )


@register_tool(
    ToolDefinition(
        name="search_web_of_science",
        description="Search papers from Web of Science Starter API.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
        ],
        tags=["search", "web-of-science"],
    )
)
def search_web_of_science(query: str, max_results: int = 10, time_range: str = "") -> List[Paper]:
    api_key = get_named_api_key("WOS_STARTER_API_KEY").strip()
    if not api_key:
        return []

    params = {
        "db": settings.wos_default_database,
        "q": _build_query(query, time_range),
        "limit": min(max(max_results * 2, 1), 50),
        "page": 1,
        "sortField": "RS+D",
    }
    headers = {"X-ApiKey": api_key}

    try:
        response = requests.get(
            settings.wos_documents_url,
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
    except Exception:
        return []

    papers: List[Paper] = []
    for item in response.json().get("hits", []):
        paper = _parse_paper(item, query)
        if not paper or not _matches_time_range(paper.year, time_range):
            continue
        papers.append(paper)
        if len(papers) >= max_results:
            break
    return papers
