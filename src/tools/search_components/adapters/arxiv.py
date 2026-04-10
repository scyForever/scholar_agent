from __future__ import annotations

from typing import List
import xml.etree.ElementTree as ET

import requests

from src.core.models import Paper
from src.tools.search_components.common import (
    ARXIV_ATOM_NS,
    DEFAULT_HEADERS,
    REQUEST_TIMEOUT,
    AcademicSourceAdapter,
    SearchRequest,
    matches_year,
    text_matches_query,
)


class ArxivAdapter(AcademicSourceAdapter):
    source_name = "arxiv"

    def search(self, request: SearchRequest) -> List[Paper]:
        search_query = request.query
        if request.author:
            search_query = f"all:{request.query} AND au:{request.author}"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max(request.max_results * 3, 1), 50),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            response = requests.get(
                "https://export.arxiv.org/api/query",
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Exception:
            return []

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            return []

        papers: List[Paper] = []
        for entry in root.findall("atom:entry", ARXIV_ATOM_NS):
            title = (entry.findtext("atom:title", default="", namespaces=ARXIV_ATOM_NS) or "").strip()
            abstract = (
                entry.findtext("atom:summary", default="", namespaces=ARXIV_ATOM_NS) or ""
            ).strip()
            if not title or not text_matches_query(title, abstract, request.query):
                continue

            published = entry.findtext("atom:published", default="", namespaces=ARXIV_ATOM_NS) or ""
            year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
            if not matches_year(year, request.time_range):
                continue

            entry_id = entry.findtext("atom:id", default="", namespaces=ARXIV_ATOM_NS) or ""
            pdf_url = ""
            html_url = entry_id
            for link in entry.findall("atom:link", ARXIV_ATOM_NS):
                title_attr = str(link.attrib.get("title") or "")
                href = str(link.attrib.get("href") or "")
                if title_attr == "pdf":
                    pdf_url = href
                    break
            authors = [
                (author.findtext("atom:name", default="", namespaces=ARXIV_ATOM_NS) or "").strip()
                for author in entry.findall("atom:author", ARXIV_ATOM_NS)
            ]
            categories = [item.attrib.get("term", "") for item in entry.findall("atom:category", ARXIV_ATOM_NS)]
            arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else ""
            papers.append(
                Paper(
                    paper_id=entry_id or arxiv_id or title,
                    title=title,
                    abstract=abstract,
                    authors=[item for item in authors if item],
                    year=year,
                    venue="arXiv",
                    url=entry_id,
                    pdf_url=pdf_url or (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""),
                    citations=0,
                    source="arXiv",
                    categories=[item for item in categories if item],
                    metadata={"updated": entry.findtext("atom:updated", default="", namespaces=ARXIV_ATOM_NS) or ""},
                    arxiv_id=arxiv_id,
                    html_url=html_url,
                    full_text_url=pdf_url or html_url,
                    open_access=True,
                )
            )
            if len(papers) >= request.max_results:
                break
        return papers
