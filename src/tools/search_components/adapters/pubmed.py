from __future__ import annotations

from typing import Dict, List
import xml.etree.ElementTree as ET

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


class PubMedAdapter(AcademicSourceAdapter):
    source_name = "pubmed"

    def search(self, request: SearchRequest) -> List[Paper]:
        ids = self._search_ids(request)
        if not ids:
            return []
        try:
            response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "xml",
                },
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
        for article in root.findall(".//PubmedArticle"):
            parsed = self._parse_article(article, request)
            if parsed is None:
                continue
            papers.append(parsed)
            if len(papers) >= request.max_results:
                break
        return papers

    def _search_ids(self, request: SearchRequest) -> List[str]:
        query = request.query
        if request.author:
            query = f"({request.query}) AND ({request.author}[Author])"
        start_year, end_year = parse_time_range(request.time_range)
        if start_year is not None and end_year is not None:
            query = f"({query}) AND ({start_year}:{end_year}[pdat])"
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": min(max(request.max_results * 3, 1), 50),
            "sort": "relevance",
        }
        api_key = get_named_api_key("NCBI_API_KEY").strip()
        if api_key:
            params["api_key"] = api_key
        try:
            response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Exception:
            return []
        payload = safe_json(response)
        return [str(item) for item in (payload.get("esearchresult") or {}).get("idlist") or [] if str(item).strip()]

    def _parse_article(self, article: ET.Element, request: SearchRequest) -> Paper | None:
        article_title = " ".join(article.findtext(".//ArticleTitle", default="").split())
        abstract_parts = [
            " ".join((node.text or "").split())
            for node in article.findall(".//Abstract/AbstractText")
            if (node.text or "").strip()
        ]
        abstract = "\n".join(item for item in abstract_parts if item)
        if not article_title or not text_matches_query(article_title, abstract, request.query):
            return None
        year = None
        for path in (".//PubDate/Year", ".//ArticleDate/Year"):
            value = article.findtext(path, default="")
            if value.isdigit():
                year = int(value)
                break
        if not matches_year(year, request.time_range):
            return None
        authors = []
        for author in article.findall(".//Author"):
            last_name = (author.findtext("LastName", default="") or "").strip()
            fore_name = (author.findtext("ForeName", default="") or "").strip()
            collective = (author.findtext("CollectiveName", default="") or "").strip()
            full_name = collective or " ".join(part for part in [fore_name, last_name] if part)
            if full_name:
                authors.append(full_name)
        journal = " ".join(article.findtext(".//Journal/Title", default="").split())
        pmid = (article.findtext(".//PMID", default="") or "").strip()
        identifiers: Dict[str, str] = {}
        for article_id in article.findall(".//ArticleId"):
            id_type = str(article_id.attrib.get("IdType") or "").strip().lower()
            value = (article_id.text or "").strip()
            if id_type and value:
                identifiers[id_type] = value
        doi = identifiers.get("doi", "")
        pmcid = identifiers.get("pmc", "")
        html_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        full_text_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else html_url
        return Paper(
            paper_id=pmid or article_title,
            title=article_title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=journal,
            url=html_url,
            pdf_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/" if pmcid else "",
            citations=0,
            source="PubMed",
            categories=["biomedical"],
            metadata={"language": article.findtext(".//Language", default="") or ""},
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            html_url=html_url,
            full_text_url=full_text_url,
            open_access=bool(pmcid),
        )
