from __future__ import annotations

from typing import Any

__all__ = ["HybridRetriever"]


def __getattr__(name: str) -> Any:
    if name == "HybridRetriever":
        from .retriever import HybridRetriever

        return HybridRetriever
    raise AttributeError(name)
