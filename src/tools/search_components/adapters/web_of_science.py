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
    matches_year,
    parse_time_range,
    safe_json,
    text_matches_query,
)


class WebOfScienceAdapter(AcademicSourceAdapter):
    source_name = "web_of_science"

    def search(self, request: SearchRequest) -> List[Paper]:
        api_key = get_named_api_key("WOS_STARTER_API_KEY").strip()
        if not api_key:
            return []
        params = {
            "db": "WOS",
            "q": self._build_query(request.query, request.author, request.time_range),
            "limit": min(max(request.max_results * 2, 1), 50),
            "page": 1,
            "sortField": "RS+D",
        }
        try:
            response = requests.get(
                "https://api.clarivate.com/apis/wos-starter/v1/documents",
                headers={"X-ApiKey": api_key, **DEFAULT_HEADERS},
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Exception:
            return []

        payload = safe_json(response)
        papers: List[Paper] = []
        for item in payload.get("hits", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            source = item.get("source") or {}
            year = int(source["publishYear"]) if source.get("publishYear") else None
            if not matches_year(year, request.time_range):
                continue
            authors = [
                str(author.get("displayName") or author.get("wosStandard") or "").strip()
                for author in (item.get("names") or {}).get("authors") or []
                if isinstance(author, dict) and (author.get("displayName") or author.get("wosStandard"))
            ]
            keywords = [
                str(keyword).strip()
                for keyword in (item.get("keywords") or {}).get("authorKeywords") or []
                if str(keyword).strip()
            ]
            if not text_matches_query(title, " ".join(keywords), request.query):
                continue
            papers.append(
                Paper(
                    paper_id=str(item.get("uid") or title),
                    title=title,
                    abstract="",
                    authors=authors,
                    year=year,
                    venue=str(source.get("sourceTitle") or ""),
                    url=str((item.get("links") or {}).get("record") or ""),
                    pdf_url="",
                    citations=max(
                        [int(citation.get("count") or 0) for citation in item.get("citations") or [] if isinstance(citation, dict)]
                        or [0]
                    ),
                    source="Web of Science",
                    categories=[str(value) for value in item.get("sourceTypes") or [] if str(value).strip()],
                    keywords=keywords,
                    metadata={"doi": (item.get("identifiers") or {}).get("doi", "")},
                    doi=str((item.get("identifiers") or {}).get("doi") or ""),
                )
            )
            if len(papers) >= request.max_results:
                break
        return papers

    def _build_query(self, query: str, author: str, time_range: str) -> str:
        clauses = [f'TS=("{query.replace(chr(34), "")}")']
        if author:
            clauses.append(f'AU=("{author.replace(chr(34), "")}")')
        start_year, end_year = parse_time_range(time_range)
        if start_year is not None and end_year is not None:
            clauses.append(f"PY={start_year}-{end_year}")
        return " AND ".join(f"({item})" for item in clauses)
