from __future__ import annotations

from typing import Any, Dict, List

from config.settings import settings
from src.rag.components.retrieval_components import (
    RAGFusionValidationComponent,
    RAGQueryPreparationComponent,
    RAGRouteRetrievalComponent,
)


class RAGRetrievalComponent:
    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self.query_preparation = RAGQueryPreparationComponent(retriever)
        self.route_retrieval = RAGRouteRetrievalComponent(retriever)
        self.fusion_validation = RAGFusionValidationComponent(retriever)

    def retrieve(
        self,
        query: str,
        chat_history: List[Dict[str, str]] | None = None,
        top_k: int | None = None,
        rewritten_queries: List[str] | None = None,
        rewrite_plan: Any = None,
    ) -> Dict[str, Any]:
        top_k = top_k or settings.rag_top_k
        prepared = self.query_preparation.prepare(
            query,
            chat_history=chat_history,
            rewritten_queries=rewritten_queries,
        )
        enhanced_query = prepared["enhanced_query"]
        rewrites = prepared["rewritten_queries"]
        routes = prepared["routes"]
        vector_enabled, vector_reason = self.retriever.vector_store.status()
        reranker_enabled, reranker_reason = self.retriever.reranker.status()
        indexed_chunk_count = self.route_retrieval.indexed_chunk_count()

        if indexed_chunk_count == 0:
            return self._empty_response(
                query=query,
                enhanced_query=enhanced_query,
                rewrites=rewrites,
                routes=routes,
                rewrite_plan=rewrite_plan,
                vector_enabled=vector_enabled,
            )
        if not vector_enabled:
            raise RuntimeError(f"Local RAG vector index is unavailable: {vector_reason}")
        if not reranker_enabled:
            raise RuntimeError(f"BGE reranker is unavailable: {reranker_reason}")
        vector_chunk_count = self.retriever.vector_store.count()
        if vector_chunk_count < indexed_chunk_count:
            raise RuntimeError(
                "Local RAG vector index is incomplete. Re-index local documents after configuring BGE-M3."
            )

        retrieval_summary = self.route_retrieval.collect_hits(rewrites, routes, top_k=top_k)
        ranked_lists = retrieval_summary["ranked_lists"]
        fused = self.fusion_validation.fuse(ranked_lists)
        reranked, validated, supplement = self.fusion_validation.rerank_and_validate(
            enhanced_query,
            fused,
            top_k=top_k,
        )

        return {
            "query": query,
            "enhanced_query": enhanced_query,
            "rewritten_queries": rewrites,
            "routes": routes,
            "results": validated[:top_k],
            "supplement": supplement,
            "trace": {
                "conversation_enhance": enhanced_query,
                "rewrites": rewrites,
                "rewrite_source": "shared_plan" if rewrite_plan is not None else "local_rag",
                "shared_core_topic": rewrite_plan.core_topic if rewrite_plan is not None else "",
                "shared_english_query": rewrite_plan.english_query if rewrite_plan is not None else "",
                "routes": routes,
                "indexed_chunk_count": indexed_chunk_count,
                "vector_chunk_count": vector_chunk_count,
                "vector_db_enabled": vector_enabled,
                "vector_db_reason": vector_reason,
                "vector_collection": settings.vector_collection_name,
                "reranker_enabled": reranker_enabled,
                "reranker_reason": reranker_reason,
                "parallel_workers": retrieval_summary["parallel_workers"],
                "parallel_tasks": sorted(
                    retrieval_summary["task_traces"],
                    key=lambda item: (str(item["rewritten_query"]), str(item["source_type"])),
                ),
                "lexical_hit_count": retrieval_summary["lexical_hit_count"],
                "vector_hit_count": retrieval_summary["vector_hit_count"],
                "before_fusion": sum(len(items) for items in ranked_lists),
                "after_rerank": len(reranked),
                "validated_count": len(validated),
            },
        }

    def _empty_response(
        self,
        *,
        query: str,
        enhanced_query: str,
        rewrites: List[str],
        routes: List[str],
        rewrite_plan: Any,
        vector_enabled: bool,
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "enhanced_query": enhanced_query,
            "rewritten_queries": rewrites,
            "routes": routes,
            "results": [],
            "supplement": [],
            "trace": {
                "conversation_enhance": enhanced_query,
                "rewrites": rewrites,
                "rewrite_source": "shared_plan" if rewrite_plan is not None else "local_rag",
                "shared_core_topic": rewrite_plan.core_topic if rewrite_plan is not None else "",
                "shared_english_query": rewrite_plan.english_query if rewrite_plan is not None else "",
                "routes": routes,
                "indexed_chunk_count": 0,
                "vector_db_enabled": vector_enabled,
                "vector_db_reason": "no_local_documents_indexed",
                "vector_collection": settings.vector_collection_name,
                "lexical_hit_count": 0,
                "vector_hit_count": 0,
                "before_fusion": 0,
                "after_rerank": 0,
                "validated_count": 0,
            },
        }
