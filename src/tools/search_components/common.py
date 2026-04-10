from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, Iterable, List, Sequence

import requests

from src.core.models import Paper


REQUEST_TIMEOUT = 30
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36 ScholarAgent/1.0"
    )
}
ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
SCHOLAR_YEAR_RE = re.compile(r"(19|20)\d{2}")
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", flags=re.IGNORECASE)


def parse_time_range(time_range: str) -> tuple[int | None, int | None]:
    normalized = str(time_range or "").strip()
    if not normalized:
        return None, None
    if normalized.startswith("last_") and normalized.endswith("_years"):
        years = int(normalized.split("_")[1])
        end_year = datetime.utcnow().year
        return end_year - years + 1, end_year
    if "-" in normalized:
        start, end = normalized.split("-", 1)
        return int(start), int(end)
    return None, None


def matches_year(year: int | None, time_range: str) -> bool:
    start_year, end_year = parse_time_range(time_range)
    if year is None or (start_year is None and end_year is None):
        return True
    if start_year is not None and year < start_year:
        return False
    if end_year is not None and year > end_year:
        return False
    return True


def extract_year(text: str) -> int | None:
    match = SCHOLAR_YEAR_RE.search(text or "")
    if not match:
        return None
    return int(match.group(0))


def normalized_terms(text: str) -> List[str]:
    return [
        token
        for token in re.split(r"[\s,.;:!?()，。；：！？、]+", str(text or "").lower())
        if len(token) > 2
    ]


def text_matches_query(title: str, abstract: str, query: str) -> bool:
    terms = normalized_terms(query)
    if not terms:
        return True
    haystack = f"{title} {abstract}".lower()
    return any(term in haystack for term in terms)


def relevance_score(paper: Paper, query: str, author: str = "") -> float:
    haystack = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}".lower()
    score = 0.0
    for term in normalized_terms(query):
        if term in haystack:
            score += 1.0
    if author:
        author_lower = author.lower()
        if any(author_lower in item.lower() for item in paper.authors):
            score += 2.0
    if paper.year:
        score += paper.year / 10000.0
    score += min(float(paper.citations or 0) / 1000.0, 3.0)
    return score


def safe_json(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def maybe_doi(text: str) -> str:
    match = DOI_RE.search(text or "")
    return match.group(0) if match else ""


def dedupe_papers(papers: Iterable[Paper], query: str, author: str = "") -> List[Paper]:
    best: Dict[str, Paper] = {}
    for paper in papers:
        key = "||".join(
            [
                (paper.doi or "").strip().lower(),
                (paper.arxiv_id or "").strip().lower(),
                (paper.title or "").strip().lower(),
            ]
        )
        if not key.strip("|"):
            continue
        paper.score = max(paper.score, relevance_score(paper, query=query, author=author))
        current = best.get(key)
        if current is None or paper.score > current.score:
            best[key] = paper
    return sorted(
        best.values(),
        key=lambda item: (item.score, item.citations, item.year or 0),
        reverse=True,
    )


@dataclass(slots=True)
class SearchRequest:
    query: str
    max_results: int = 10
    time_range: str = ""
    author: str = ""
    platforms: Sequence[str] = field(default_factory=list)


class AcademicSourceAdapter:
    source_name: str = ""

    def search(self, request: SearchRequest) -> List[Paper]:
        raise NotImplementedError
