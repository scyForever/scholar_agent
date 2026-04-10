from __future__ import annotations

from typing import Any, Dict, List

import requests

from api_keys import get_named_api_key
from src.core.models import Paper
from src.tools.search_components.common import (
    DEFAULT_HEADERS,
    REQUEST_TIMEOUT,
    AcademicSourceAdapter,
    SearchRequest,
    extract_year,
    matches_year,
    parse_time_range,
    safe_json,
    text_matches_query,
)


class IEEEXploreAdapter(AcademicSourceAdapter):
    source_name = "ieee_xplore"

    def search(self, request: SearchRequest) -> List[Paper]:
        api_key = get_named_api_key("IEEE_XPLORE_API_KEY").strip()
        if not api_key:
            return []
        params: Dict[str, Any] = {
            "apikey": api_key,
            "querytext": request.query,
            "max_records": min(max(request.max_results * 3, 1), 50),
            "start_record": 1,
            "format": "json",
            "sort_order": "desc",
            "sort_field": "article_number",
        }
        if request.author:
            params["author"] = request.author
        start_year, end_year = parse_time_range(request.time_range)
        if start_year is not None:
            params["start_year"] = start_year
        if end_year is not None:
            params["end_year"] = end_year
        try:
            response = requests.get(
                "https://ieeexploreapi.ieee.org/api/v1/search/articles",
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Exception:
            return []
        payload = safe_json(response)
        papers: List[Paper] = []
        for item in payload.get("articles", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            abstract = str(item.get("abstract") or "").strip()
            if not title or not text_matches_query(title, abstract, request.query):
                continue
            year = int(item["publication_year"]) if item.get("publication_year") else extract_year(str(item.get("publication_date") or ""))
            if not matches_year(year, request.time_range):
                continue
            papers.append(
                Paper(
                    paper_id=str(item.get("article_number") or title),
                    title=title,
                    abstract=abstract,
                    authors=[
                        str(author.get("full_name") or "").strip()
                        for author in (item.get("authors") or {}).get("authors") or []
                        if isinstance(author, dict) and author.get("full_name")
                    ],
                    year=year,
                    venue=str(item.get("publication_title") or ""),
                    url=str(item.get("html_url") or item.get("abstract_url") or ""),
                    pdf_url=str(item.get("pdf_url") or ""),
                    citations=int(item.get("citing_paper_count") or 0),
                    source="IEEE Xplore",
                    categories=[
                        str(item.get("content_type") or "").strip(),
                        *[
                            str(term).strip()
                            for term in (item.get("index_terms") or {}).get("ieee_terms", {}).get("terms") or []
                            if str(term).strip()
                        ],
                    ],
                    keywords=[
                        str(term).strip()
                        for term in (item.get("index_terms") or {}).get("author_terms", {}).get("terms") or []
                        if str(term).strip()
                    ],
                    metadata={"access_type": item.get("access_type", "")},
                    doi=str(item.get("doi") or ""),
                    html_url=str(item.get("html_url") or ""),
                    full_text_url=str(item.get("html_url") or ""),
                    open_access=str(item.get("access_type") or "").lower() == "open access",
                )
            )
            if len(papers) >= request.max_results:
                break
        return papers
