from __future__ import annotations

from datetime import datetime
from typing import List

from src.core.models import Paper
from src.preprocessing.query_rewriter import QueryRewriter
from src.tools.registry import ToolDefinition, ToolParameter, register_tool


TOPIC_CATEGORIES = {
    "reinforcement learning": ["cs.LG", "cs.AI"],
    "multi-agent reinforcement learning": ["cs.LG", "cs.AI"],
    "graph neural network": ["cs.LG", "cs.AI"],
    "retrieval augmented generation": ["cs.CL", "cs.IR"],
    "large language model": ["cs.CL", "cs.AI"],
}


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


def _is_relevant(title: str, abstract: str, query: str) -> bool:
    query_terms = [term for term in query.lower().replace("/", " ").split() if len(term) > 2]
    text = f"{title} {abstract}".lower()
    return not query_terms or any(term in text for term in query_terms)


@register_tool(
    ToolDefinition(
        name="search_arxiv",
        description="Search papers from arXiv.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
        ],
        tags=["search", "arxiv"],
    )
)
def search_arxiv(query: str, max_results: int = 10, time_range: str = "") -> List[Paper]:
    try:
        import arxiv
    except ImportError:
        return []

    rewritten = QueryRewriter().to_english_query(query)
    lowered = rewritten.lower()
    category_filter = []
    for topic, categories in TOPIC_CATEGORIES.items():
        if topic in lowered:
            category_filter = categories
            break

    search = arxiv.Search(
        query=rewritten,
        max_results=max_results * 3,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers: List[Paper] = []
    try:
        for result in search.results():
            year = result.published.year if result.published else None
            categories = list(result.categories or [])
            if category_filter and not set(categories).intersection(category_filter):
                continue
            if not _matches_time_range(year, time_range):
                continue
            if not _is_relevant(result.title, result.summary, rewritten):
                continue
            papers.append(
                Paper(
                    paper_id=result.entry_id,
                    title=result.title.strip(),
                    abstract=(result.summary or "").strip(),
                    authors=[author.name for author in result.authors],
                    year=year,
                    venue="arXiv",
                    url=result.entry_id,
                    pdf_url=result.pdf_url or "",
                    citations=0,
                    source="arXiv",
                    categories=categories,
                    keywords=[],
                    metadata={"updated": result.updated.isoformat() if result.updated else ""},
                )
            )
            if len(papers) >= max_results:
                break
    except Exception:
        return []
    return papers
