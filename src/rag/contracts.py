from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.preprocessing.query_rewriter import QueryRewritePlan


@dataclass(slots=True)
class RAGIndexPDFRequest:
    pdf_path: str
    title: str | None = None
    metadata: Dict[str, Any] | None = None


@dataclass(slots=True)
class RAGIndexTextRequest:
    title: str
    text: str
    metadata: Dict[str, Any] | None = None


@dataclass(slots=True)
class RAGRetrieveRequest:
    query: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    top_k: int | None = None
    rewritten_queries: List[str] | None = None
    rewrite_plan: QueryRewritePlan | None = None
