from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from src.rag.contracts import RAGIndexPDFRequest, RAGIndexTextRequest, RAGRetrieveRequest

if TYPE_CHECKING:
    from src.rag.retriever import HybridRetriever


class RAGIndexingHarness:
    def __init__(self, retriever: "HybridRetriever") -> None:
        self.retriever = retriever

    def index_pdf(self, request: RAGIndexPDFRequest) -> str:
        return self.retriever._index_pdf_impl(
            request.pdf_path,
            title=request.title,
            metadata=request.metadata,
        )

    def index_text(self, request: RAGIndexTextRequest) -> str:
        return self.retriever._index_text_impl(
            request.title,
            request.text,
            metadata=request.metadata,
        )


class RAGRetrievalHarness:
    def __init__(self, retriever: "HybridRetriever") -> None:
        self.retriever = retriever

    def retrieve(self, request: RAGRetrieveRequest) -> Dict[str, Any]:
        return self.retriever._retrieve_impl(
            request.query,
            chat_history=request.chat_history,
            top_k=request.top_k,
            rewritten_queries=request.rewritten_queries,
            rewrite_plan=request.rewrite_plan,
        )


class RAGHarness:
    """RAG 层统一入口 harness，拆分索引与检索子能力。"""

    def __init__(self, retriever: "HybridRetriever") -> None:
        self.retriever = retriever
        self.indexing = RAGIndexingHarness(retriever)
        self.retrieval = RAGRetrievalHarness(retriever)

    def index_pdf(self, request: RAGIndexPDFRequest) -> str:
        return self.indexing.index_pdf(request)

    def index_text(self, request: RAGIndexTextRequest) -> str:
        return self.indexing.index_text(request)

    def retrieve(self, request: RAGRetrieveRequest) -> Dict[str, Any]:
        return self.retrieval.retrieve(request)
