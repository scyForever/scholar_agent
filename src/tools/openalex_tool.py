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


def _parse_work(work: Dict[str, Any], search_query: str) -> Optional[Paper]:
    if not work or not isinstance(work, dict):
        return None

    title = work.get("title") or ""
    if not title:
        return None

    abstract = work.get("abstract_inverted_index") or {}
    if abstract:
        ordered = sorted(
            ((position, token) for token, positions in abstract.items() for position in positions),
            key=lambda item: item[0],
        )
        abstract_text = " ".join(token for _, token in ordered)
    else:
        abstract_text = ""

    text = f"{title} {abstract_text}".lower()
    query_terms = [term for term in search_query.lower().split() if len(term) > 2]
    if query_terms and not any(term in text for term in query_terms):
        return None

    authors: List[str] = []
    authorships = work.get("authorships")
    if authorships and isinstance(authorships, list):
        for authorship in authorships:
            if authorship and isinstance(authorship, dict):
                author = authorship.get("author")
                if author and isinstance(author, dict):
                    name = author.get("display_name")
                    if name:
                        authors.append(str(name))

    primary_location = work.get("primary_location") or {}
    landing_page_url = primary_location.get("landing_page_url") or work.get("id") or ""
    pdf_url = primary_location.get("pdf_url") or ""
    year = work.get("publication_year")

    return Paper(
        paper_id=str(work.get("id") or landing_page_url),
        title=title,
        abstract=abstract_text,
        authors=authors,
        year=int(year) if year else None,
        venue=str((work.get("host_venue") or {}).get("display_name") or ""),
        url=landing_page_url,
        pdf_url=pdf_url,
        citations=int(work.get("cited_by_count") or 0),
        source="OpenAlex",
        categories=[str(item.get("display_name")) for item in work.get("concepts") or [] if item.get("display_name")],
        keywords=[],
        metadata={"type": work.get("type", "")},
    )


@register_tool(
    ToolDefinition(
        name="search_openalex",
        description="Search papers from OpenAlex.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
        ],
        tags=["search", "openalex"],
    )
)
def search_openalex(query: str, max_results: int = 10, time_range: str = "") -> List[Paper]:
    rewritten = QueryRewriter().normalize_topic(query)
    try:
        response = requests.get(
            "https://api.openalex.org/works",
            params={"search": rewritten, "per-page": max_results * 3},
            timeout=30,
        )
        response.raise_for_status()
    except Exception:
        return []

    papers: List[Paper] = []
    for work in response.json().get("results", []):
        paper = _parse_work(work, rewritten)
        if not paper or not _matches_time_range(paper.year, time_range):
            continue
        papers.append(paper)
        if len(papers) >= max_results:
            break
    return papers
