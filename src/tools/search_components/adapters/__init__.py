from __future__ import annotations

from .arxiv import ArxivAdapter
from .google_scholar import GoogleScholarAdapter
from .ieee_xplore import IEEEXploreAdapter
from .openalex import OpenAlexAdapter
from .pubmed import PubMedAdapter
from .semantic_scholar import SemanticScholarAdapter
from .web_of_science import WebOfScienceAdapter

__all__ = [
    "ArxivAdapter",
    "OpenAlexAdapter",
    "SemanticScholarAdapter",
    "WebOfScienceAdapter",
    "PubMedAdapter",
    "IEEEXploreAdapter",
    "GoogleScholarAdapter",
]
