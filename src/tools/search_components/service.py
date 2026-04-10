from __future__ import annotations

from typing import Dict, List

from src.core.models import Paper
from src.preprocessing.query_rewriter import QueryRewriter
from src.tools.search_components.adapters import (
    ArxivAdapter,
    GoogleScholarAdapter,
    IEEEXploreAdapter,
    OpenAlexAdapter,
    PubMedAdapter,
    SemanticScholarAdapter,
    WebOfScienceAdapter,
)
from src.tools.search_components.common import SearchRequest, dedupe_papers


class LiteratureSearchService:
    def __init__(self) -> None:
        self.adapters: Dict[str, object] = {
            "arxiv": ArxivAdapter(),
            "openalex": OpenAlexAdapter(),
            "semantic_scholar": SemanticScholarAdapter(),
            "web_of_science": WebOfScienceAdapter(),
            "pubmed": PubMedAdapter(),
            "ieee_xplore": IEEEXploreAdapter(),
            "google_scholar": GoogleScholarAdapter(),
        }
        self.rewriter = QueryRewriter()

    def search(self, request: SearchRequest) -> Dict[str, object]:
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
        ranked = dedupe_papers(papers, query=rewritten, author=request.author)
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
        return dedupe_papers(papers, query=rewritten, author=request.author)[: request.max_results]

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
