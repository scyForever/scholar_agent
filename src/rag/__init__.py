from __future__ import annotations

from typing import Any

__all__ = ["HybridRetriever", "RAGHarness", "RAGIndexingHarness", "RAGRetrievalHarness"]


def __getattr__(name: str) -> Any:
    if name == "HybridRetriever":
        from .retriever import HybridRetriever

        return HybridRetriever
    if name in {"RAGHarness", "RAGIndexingHarness", "RAGRetrievalHarness"}:
        from .harness import RAGHarness, RAGIndexingHarness, RAGRetrievalHarness

        mapping = {
            "RAGHarness": RAGHarness,
            "RAGIndexingHarness": RAGIndexingHarness,
            "RAGRetrievalHarness": RAGRetrievalHarness,
        }
        return mapping[name]
    raise AttributeError(name)
