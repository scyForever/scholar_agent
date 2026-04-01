from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, Iterable, List, Sequence
import xml.etree.ElementTree as ET

import requests

from api_keys import get_named_api_key
from src.core.models import Paper
from src.preprocessing.query_rewriter import QueryRewriter
from src.tools.registry import ToolDefinition, ToolParameter, register_tool

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


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


def _parse_time_range(time_range: str) -> tuple[int | None, int | None]:
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


def _matches_year(year: int | None, time_range: str) -> bool:
    start_year, end_year = _parse_time_range(time_range)
    if year is None or (start_year is None and end_year is None):
        return True
    if start_year is not None and year < start_year:
        return False
    if end_year is not None and year > end_year:
        return False
    return True


def _extract_year(text: str) -> int | None:
    match = SCHOLAR_YEAR_RE.search(text or "")
    if not match:
        return None
    return int(match.group(0))


def _normalized_terms(text: str) -> List[str]:
    return [
        token
        for token in re.split(r"[\s,.;:!?()，。；：！？、]+", str(text or "").lower())
        if len(token) > 2
    ]


def _text_matches_query(title: str, abstract: str, query: str) -> bool:
    terms = _normalized_terms(query)
    if not terms:
        return True
    haystack = f"{title} {abstract}".lower()
    return any(term in haystack for term in terms)


def _relevance_score(paper: Paper, query: str, author: str = "") -> float:
    haystack = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}".lower()
    score = 0.0
    for term in _normalized_terms(query):
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


def _safe_json(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _maybe_doi(text: str) -> str:
    match = DOI_RE.search(text or "")
    return match.group(0) if match else ""


def _dedupe_papers(papers: Iterable[Paper], query: str, author: str = "") -> List[Paper]:
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
        paper.score = max(paper.score, _relevance_score(paper, query=query, author=author))
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
            if not title or not _text_matches_query(title, abstract, request.query):
                continue

            published = entry.findtext("atom:published", default="", namespaces=ARXIV_ATOM_NS) or ""
            year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
            if not _matches_year(year, request.time_range):
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
        start_year, end_year = _parse_time_range(request.time_range)
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

        payload = _safe_json(response)
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
            if not _text_matches_query(title, abstract, request.query):
                continue
            year = int(work["publication_year"]) if work.get("publication_year") else None
            if not _matches_year(year, request.time_range):
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

        payload = _safe_json(response)
        papers: List[Paper] = []
        for item in payload.get("data", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            abstract = str(item.get("abstract") or "").strip()
            if not title or not _text_matches_query(title, abstract, request.query):
                continue
            year = int(item["year"]) if item.get("year") else None
            if not _matches_year(year, request.time_range):
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

        payload = _safe_json(response)
        papers: List[Paper] = []
        for item in payload.get("hits", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            source = item.get("source") or {}
            year = int(source["publishYear"]) if source.get("publishYear") else None
            if not _matches_year(year, request.time_range):
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
            if not _text_matches_query(title, " ".join(keywords), request.query):
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
        start_year, end_year = _parse_time_range(time_range)
        if start_year is not None and end_year is not None:
            clauses.append(f"PY={start_year}-{end_year}")
        return " AND ".join(f"({item})" for item in clauses)


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
        start_year, end_year = _parse_time_range(request.time_range)
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
        payload = _safe_json(response)
        return [str(item) for item in (payload.get("esearchresult") or {}).get("idlist") or [] if str(item).strip()]

    def _parse_article(self, article: ET.Element, request: SearchRequest) -> Paper | None:
        article_title = " ".join(article.findtext(".//ArticleTitle", default="").split())
        abstract_parts = [
            " ".join((node.text or "").split())
            for node in article.findall(".//Abstract/AbstractText")
            if (node.text or "").strip()
        ]
        abstract = "\n".join(item for item in abstract_parts if item)
        if not article_title or not _text_matches_query(article_title, abstract, request.query):
            return None
        year = None
        for path in (".//PubDate/Year", ".//ArticleDate/Year"):
            value = article.findtext(path, default="")
            if value.isdigit():
                year = int(value)
                break
        if not _matches_year(year, request.time_range):
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
        start_year, end_year = _parse_time_range(request.time_range)
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
        payload = _safe_json(response)
        papers: List[Paper] = []
        for item in payload.get("articles", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            abstract = str(item.get("abstract") or "").strip()
            if not title or not _text_matches_query(title, abstract, request.query):
                continue
            year = int(item["publication_year"]) if item.get("publication_year") else _extract_year(str(item.get("publication_date") or ""))
            if not _matches_year(year, request.time_range):
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
            year = _extract_year(meta_text)
            if not _matches_year(year, request.time_range):
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
            if not _text_matches_query(title, snippet, request.query):
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
                    doi=_maybe_doi(snippet),
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
        payload = _safe_json(response)
        papers: List[Paper] = []
        for item in payload.get("organic_results", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            abstract = str(item.get("snippet") or "").strip()
            publication_info = item.get("publication_info") or {}
            summary = str(publication_info.get("summary") or "")
            year = _extract_year(summary)
            if not title or not _matches_year(year, request.time_range):
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
                    doi=_maybe_doi(abstract) or _maybe_doi(summary),
                    full_text_url=pdf_url or str(item.get("link") or ""),
                    html_url=str(item.get("link") or ""),
                    open_access=bool(pdf_url),
                )
            )
            if len(papers) >= request.max_results:
                break
        return papers


class LiteratureSearchService:
    def __init__(self) -> None:
        self.adapters: Dict[str, AcademicSourceAdapter] = {
            "arxiv": ArxivAdapter(),
            "openalex": OpenAlexAdapter(),
            "semantic_scholar": SemanticScholarAdapter(),
            "web_of_science": WebOfScienceAdapter(),
            "pubmed": PubMedAdapter(),
            "ieee_xplore": IEEEXploreAdapter(),
            "google_scholar": GoogleScholarAdapter(),
        }
        self.rewriter = QueryRewriter()

    def search(self, request: SearchRequest) -> Dict[str, Any]:
        rewritten = self.rewriter.to_english_query(request.query)
        normalized_platforms = [self._normalize_platform(item) for item in request.platforms if self._normalize_platform(item)]
        selected_platforms = normalized_platforms or list(self.adapters)
        papers: List[Paper] = []
        source_breakdown: Dict[str, int] = {}
        for platform in selected_platforms:
            adapter = self.adapters.get(platform)
            if adapter is None:
                continue
            source_papers = adapter.search(
                SearchRequest(
                    query=rewritten,
                    max_results=request.max_results,
                    time_range=request.time_range,
                    author=request.author,
                    platforms=[platform],
                )
            )
            papers.extend(source_papers)
            source_breakdown[platform] = len(source_papers)
        ranked = _dedupe_papers(papers, query=rewritten, author=request.author)
        return {
            "query": request.query,
            "rewritten_query": rewritten,
            "platforms": selected_platforms,
            "source_breakdown": source_breakdown,
            "papers": ranked[: request.max_results],
        }

    def search_by_source(self, source_name: str, request: SearchRequest) -> List[Paper]:
        adapter = self.adapters.get(self._normalize_platform(source_name))
        if adapter is None:
            return []
        rewritten = self.rewriter.to_english_query(request.query)
        papers = adapter.search(
            SearchRequest(
                query=rewritten,
                max_results=request.max_results,
                time_range=request.time_range,
                author=request.author,
                platforms=[source_name],
            )
        )
        return _dedupe_papers(papers, query=rewritten, author=request.author)[: request.max_results]

    def _normalize_platform(self, value: str) -> str:
        normalized = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "arxiv": "arxiv",
            "search_arxiv": "arxiv",
            "openalex": "openalex",
            "search_openalex": "openalex",
            "semantic_scholar": "semantic_scholar",
            "semanticscholar": "semantic_scholar",
            "search_semantic_scholar": "semantic_scholar",
            "wos": "web_of_science",
            "web_of_science": "web_of_science",
            "search_web_of_science": "web_of_science",
            "pubmed": "pubmed",
            "pm": "pubmed",
            "search_pubmed": "pubmed",
            "google_scholar": "google_scholar",
            "scholar": "google_scholar",
            "search_google_scholar": "google_scholar",
            "ieee": "ieee_xplore",
            "ieee_xplore": "ieee_xplore",
            "search_ieee_xplore": "ieee_xplore",
        }
        return aliases.get(normalized, normalized)


SEARCH_SERVICE = LiteratureSearchService()


def search_source(source_name: str, query: str, max_results: int = 10, time_range: str = "", author: str = "") -> List[Paper]:
    return SEARCH_SERVICE.search_by_source(
        source_name,
        SearchRequest(query=query, max_results=max_results, time_range=time_range, author=author),
    )


@register_tool(
    ToolDefinition(
        name="search_google_scholar",
        description="Search papers from Google Scholar.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
        ],
        tags=["search", "google-scholar"],
    )
)
def search_google_scholar(query: str, max_results: int = 10, time_range: str = "") -> List[Paper]:
    return search_source("google_scholar", query, max_results=max_results, time_range=time_range)


@register_tool(
    ToolDefinition(
        name="search_pubmed",
        description="Search papers from PubMed.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
        ],
        tags=["search", "pubmed"],
    )
)
def search_pubmed(query: str, max_results: int = 10, time_range: str = "") -> List[Paper]:
    return search_source("pubmed", query, max_results=max_results, time_range=time_range)


@register_tool(
    ToolDefinition(
        name="search_ieee_xplore",
        description="Search papers from IEEE Xplore.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
        ],
        tags=["search", "ieee-xplore"],
    )
)
def search_ieee_xplore(query: str, max_results: int = 10, time_range: str = "") -> List[Paper]:
    return search_source("ieee_xplore", query, max_results=max_results, time_range=time_range)


@register_tool(
    ToolDefinition(
        name="search_literature",
        description="Search papers across multiple academic platforms.",
        parameters=[
            ToolParameter("query", "str", "搜索关键词", True),
            ToolParameter("platforms", "list", "检索平台列表", False),
            ToolParameter("max_results", "int", "返回论文数量", False),
            ToolParameter("time_range", "str", "时间范围", False),
            ToolParameter("author", "str", "作者名", False),
        ],
        tags=["search", "literature"],
    )
)
def search_literature(
    query: str,
    platforms: List[str] | None = None,
    max_results: int = 10,
    time_range: str = "",
    author: str = "",
) -> Dict[str, Any]:
    result = SEARCH_SERVICE.search(
        SearchRequest(
            query=query,
            platforms=platforms or [],
            max_results=max_results,
            time_range=time_range,
            author=author,
        )
    )
    return {
        **result,
        "papers": [asdict(paper) for paper in result["papers"]],
    }


def search_platforms(
    query: str,
    *,
    platforms: Sequence[str] | None = None,
    max_results: int = 10,
    time_range: str = "",
    author: str = "",
) -> Dict[str, Any]:
    return SEARCH_SERVICE.search(
        SearchRequest(
            query=query,
            platforms=platforms or [],
            max_results=max_results,
            time_range=time_range,
            author=author,
        )
    )
