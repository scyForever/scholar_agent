from __future__ import annotations

from typing import List

import requests

from src.core.models import Paper
from src.tools.search_components.common import (
    DEFAULT_HEADERS,
    REQUEST_TIMEOUT,
    AcademicSourceAdapter,
    SearchRequest,
    matches_year,
    safe_json,
    text_matches_query,
)


class SemanticScholarAdapter(AcademicSourceAdapter):
    source_name = "semantic_scholar"

    def search(self, request: SearchRequest) -> List[Paper]:
        params = {
            "query": request.query,
            "limit": min(max(request.max_results * 3, 1), 50),
            "fields": (
                "paperId,title,abstract,authors,year,venue,url,openAccessPdf,"
                "citationCount,influentialCitationCount,fieldsOfStudy,externalIds"
            ),
        }
        try:
            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Exception:
            return []

        payload = safe_json(response)
        papers: List[Paper] = []
        for item in payload.get("data", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            abstract = str(item.get("abstract") or "").strip()
            if not title or not text_matches_query(title, abstract, request.query):
                continue
            year = int(item["year"]) if item.get("year") else None
            if not matches_year(year, request.time_range):
                continue
            external_ids = item.get("externalIds") or {}
            papers.append(
                Paper(
                    paper_id=str(item.get("paperId") or title),
                    title=title,
                    abstract=abstract,
                    authors=[
                        str(author.get("name") or "").strip()
                        for author in item.get("authors") or []
                        if isinstance(author, dict) and author.get("name")
                    ],
                    year=year,
                    venue=str(item.get("venue") or ""),
                    url=str(item.get("url") or ""),
                    pdf_url=str(((item.get("openAccessPdf") or {}).get("url")) or ""),
                    citations=int(item.get("citationCount") or 0),
                    source="Semantic Scholar",
                    categories=[
                        str(field.get("category") or field)
                        for field in item.get("fieldsOfStudy") or []
                        if str(field).strip()
                    ],
                    metadata={
                        "influential_citation_count": item.get("influentialCitationCount", 0)
                    },
                    doi=str(external_ids.get("DOI") or ""),
                    arxiv_id=str(external_ids.get("ArXiv") or ""),
                    open_access=bool((item.get("openAccessPdf") or {}).get("url")),
                    full_text_url=str(((item.get("openAccessPdf") or {}).get("url")) or ""),
                )
            )
            if len(papers) >= request.max_results:
                break
        return papers
