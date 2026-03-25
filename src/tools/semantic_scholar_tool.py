from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from src.core.models import Paper
from src.preprocessing.query_rewriter import QueryRewriter
from src.tools.registry import ToolDefinition, ToolParameter, register_tool


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


def _parse_paper(item: Dict[str, Any], query: str) -> Optional[Paper]:
    if not item or not isinstance(item, dict):
        return None
    title = item.get("title") or ""
    abstract = item.get("abstract") or ""
    if not title:
        return None
    text = f"{title} {abstract}".lower()
    terms = [term for term in query.lower().split() if len(term) > 2]
    if terms and not any(term in text for term in terms):
        return None
    authors = [author.get("name", "") for author in item.get("authors") or [] if author.get("name")]
    external_ids = item.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv", "")
    fields = []
    for field in item.get("fieldsOfStudy") or []:
        if isinstance(field, dict) and field.get("category"):
            fields.append(str(field["category"]))
        elif isinstance(field, str):
            fields.append(field)

    return Paper(
        paper_id=str(item.get("paperId") or arxiv_id or title),
        title=title,
        abstract=abstract,
        authors=authors,
        year=item.get("year"),
        venue=item.get("venue") or "",
        url=item.get("url") or "",
        pdf_url=((item.get("openAccessPdf") or {}).get("url") or ""),
        citations=int(item.get("citationCount") or 0),
        source="Semantic Scholar",
        categories=fields,
        keywords=[],
        metadata={"influential_citation_count": item.get("influentialCitationCount", 0)},
    )


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
    rewritten = QueryRewriter().normalize_topic(query)
    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": rewritten,
                "limit": max_results * 3,
                "fields": "paperId,title,abstract,authors,year,venue,url,openAccessPdf,citationCount,influentialCitationCount,fieldsOfStudy,externalIds",
            },
            timeout=30,
        )
        response.raise_for_status()
    except Exception:
        return []

    papers: List[Paper] = []
    for item in response.json().get("data", []):
        paper = _parse_paper(item, rewritten)
        if not paper or not _matches_time_range(paper.year, time_range):
            continue
        papers.append(paper)
        if len(papers) >= max_results:
            break
    return papers
