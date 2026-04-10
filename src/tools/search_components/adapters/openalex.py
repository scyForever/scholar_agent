from __future__ import annotations

from typing import Any, Dict, List

import requests

from src.core.models import Paper
from src.tools.search_components.common import (
    DEFAULT_HEADERS,
    REQUEST_TIMEOUT,
    AcademicSourceAdapter,
    SearchRequest,
    matches_year,
    parse_time_range,
    safe_json,
    text_matches_query,
)


class OpenAlexAdapter(AcademicSourceAdapter):
    source_name = "openalex"

    def search(self, request: SearchRequest) -> List[Paper]:
        params: Dict[str, Any] = {
            "search": request.query,
            "per-page": min(max(request.max_results * 3, 1), 50),
        }
        filters: List[str] = []
        if request.author:
            filters.append(f"authorships.author.display_name.search:{request.author}")
        start_year, end_year = parse_time_range(request.time_range)
        if start_year is not None:
            filters.append(f"from_publication_date:{start_year}-01-01")
        if end_year is not None:
            filters.append(f"to_publication_date:{end_year}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        try:
            response = requests.get(
                "https://api.openalex.org/works",
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Exception:
            return []

        payload = safe_json(response)
        papers: List[Paper] = []
        for work in payload.get("results", []):
            if not isinstance(work, dict):
                continue
            title = str(work.get("title") or "").strip()
            if not title:
                continue
            abstract_index = work.get("abstract_inverted_index") or {}
            abstract = ""
            if isinstance(abstract_index, dict):
                ordered = sorted(
                    ((position, token) for token, positions in abstract_index.items() for position in positions),
                    key=lambda item: item[0],
                )
                abstract = " ".join(token for _, token in ordered)
            if not text_matches_query(title, abstract, request.query):
                continue
            year = int(work["publication_year"]) if work.get("publication_year") else None
            if not matches_year(year, request.time_range):
                continue
            location = work.get("primary_location") or {}
            source = location.get("source") or {}
            authors: List[str] = []
            for authorship in work.get("authorships") or []:
                if not isinstance(authorship, dict):
                    continue
                author = authorship.get("author") or {}
                name = str(author.get("display_name") or "").strip()
                if name:
                    authors.append(name)
            papers.append(
                Paper(
                    paper_id=str(work.get("id") or title),
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    year=year,
                    venue=str(source.get("display_name") or ""),
                    url=str(location.get("landing_page_url") or work.get("id") or ""),
                    pdf_url=str(location.get("pdf_url") or ""),
                    citations=int(work.get("cited_by_count") or 0),
                    source="OpenAlex",
                    categories=[
                        str(item.get("display_name"))
                        for item in work.get("concepts") or []
                        if isinstance(item, dict) and item.get("display_name")
                    ],
                    metadata={"type": work.get("type", "")},
                    doi=str(work.get("doi") or "").replace("https://doi.org/", ""),
                    open_access=bool((work.get("open_access") or {}).get("is_oa")),
                    full_text_url=str(location.get("pdf_url") or location.get("landing_page_url") or ""),
                    html_url=str(location.get("landing_page_url") or ""),
                )
            )
            if len(papers) >= request.max_results:
                break
        return papers
