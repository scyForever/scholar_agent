from __future__ import annotations

from typing import List

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
    maybe_doi,
    safe_json,
    text_matches_query,
)

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


class GoogleScholarAdapter(AcademicSourceAdapter):
    source_name = "google_scholar"

    def search(self, request: SearchRequest) -> List[Paper]:
        papers = self._search_serpapi(request)
        if papers:
            return papers
        if BeautifulSoup is None:
            return []
        try:
            response = requests.get(
                "https://scholar.google.com/scholar",
                params={"q": request.query, "hl": "en"},
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Exception:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        papers: List[Paper] = []
        for result in soup.select("div.gs_ri"):
            title_node = result.select_one("h3.gs_rt")
            snippet_node = result.select_one("div.gs_rs")
            meta_node = result.select_one("div.gs_a")
            pdf_node = result.find_previous("div", class_="gs_or_ggsm")
            if title_node is None:
                continue
            title = title_node.get_text(" ", strip=True)
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            meta_text = meta_node.get_text(" ", strip=True) if meta_node else ""
            year = extract_year(meta_text)
            if not matches_year(year, request.time_range):
                continue
            authors = [item.strip() for item in meta_text.split(" - ")[0].split(",") if item.strip()]
            cited_by = 0
            for link in result.select("a"):
                label = link.get_text(" ", strip=True)
                if label.lower().startswith("cited by"):
                    try:
                        cited_by = int(label.split()[-1])
                    except Exception:
                        cited_by = 0
                    break
            href = ""
            link_node = title_node.find("a")
            if link_node is not None:
                href = str(link_node.get("href") or "")
            pdf_url = ""
            if pdf_node is not None:
                anchor = pdf_node.find("a")
                if anchor is not None:
                    pdf_url = str(anchor.get("href") or "")
            if not text_matches_query(title, snippet, request.query):
                continue
            papers.append(
                Paper(
                    paper_id=href or title,
                    title=title,
                    abstract=snippet,
                    authors=authors,
                    year=year,
                    venue="Google Scholar",
                    url=href,
                    pdf_url=pdf_url,
                    citations=cited_by,
                    source="Google Scholar",
                    metadata={"scholar_meta": meta_text},
                    doi=maybe_doi(snippet),
                    full_text_url=pdf_url or href,
                    html_url=href,
                    open_access=bool(pdf_url),
                )
            )
            if len(papers) >= request.max_results:
                break
        return papers

    def _search_serpapi(self, request: SearchRequest) -> List[Paper]:
        api_key = get_named_api_key("SERPAPI_API_KEY").strip()
        if not api_key:
            return []
        try:
            response = requests.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google_scholar",
                    "q": request.query,
                    "api_key": api_key,
                    "num": min(max(request.max_results * 2, 1), 20),
                    "hl": "en",
                },
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Exception:
            return []
        payload = safe_json(response)
        papers: List[Paper] = []
        for item in payload.get("organic_results", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            abstract = str(item.get("snippet") or "").strip()
            publication_info = item.get("publication_info") or {}
            summary = str(publication_info.get("summary") or "")
            year = extract_year(summary)
            if not title or not matches_year(year, request.time_range):
                continue
            authors = [
                str(author.get("name") or "").strip()
                for author in publication_info.get("authors") or []
                if isinstance(author, dict) and author.get("name")
            ]
            inline_links = item.get("inline_links") or {}
            cited_by_total = int((((inline_links.get("cited_by") or {}).get("total")) or 0))
            resources = item.get("resources") or []
            pdf_url = ""
            for resource in resources:
                if not isinstance(resource, dict):
                    continue
                link = str(resource.get("link") or "")
                if link.lower().endswith(".pdf"):
                    pdf_url = link
                    break
            papers.append(
                Paper(
                    paper_id=str(item.get("result_id") or title),
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    year=year,
                    venue="Google Scholar",
                    url=str(item.get("link") or ""),
                    pdf_url=pdf_url,
                    citations=cited_by_total,
                    source="Google Scholar",
                    metadata={"publication_summary": summary},
                    doi=maybe_doi(abstract) or maybe_doi(summary),
                    full_text_url=pdf_url or str(item.get("link") or ""),
                    html_url=str(item.get("link") or ""),
                    open_access=bool(pdf_url),
                )
            )
            if len(papers) >= request.max_results:
                break
        return papers
