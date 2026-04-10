from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from src.core.models import IndexedChunk
from src.rag.components.common import bm25_scores


class RAGRouteRetrievalComponent:
    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def indexed_chunk_count(self) -> int:
        with self.retriever._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"] if row is not None else 0)

    def collect_hits(
        self,
        rewritten_queries: List[str],
        routes: List[str],
        *,
        top_k: int,
    ) -> Dict[str, Any]:
        ranked_lists: List[List[IndexedChunk]] = []
        lexical_hit_count = 0
        vector_hit_count = 0
        task_traces: List[Dict[str, Any]] = []
        tasks = [(rewritten_query, source_type) for rewritten_query in rewritten_queries for source_type in routes]
        max_workers = max(1, min(settings.rag_parallel_workers, len(tasks)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._retrieve_route, rewritten_query, source_type, top_k * 2)
                for rewritten_query, source_type in tasks
            ]
            for future in as_completed(futures):
                route_result = future.result()
                lexical_hits = route_result["lexical_hits"]
                semantic_hits = route_result["semantic_hits"]
                if lexical_hits:
                    ranked_lists.append(lexical_hits)
                    lexical_hit_count += len(lexical_hits)
                if semantic_hits:
                    ranked_lists.append(semantic_hits)
                    vector_hit_count += len(semantic_hits)
                task_traces.append(
                    {
                        "rewritten_query": route_result["rewritten_query"],
                        "source_type": route_result["source_type"],
                        "lexical_hits": len(lexical_hits),
                        "vector_hits": len(semantic_hits),
                    }
                )
        return {
            "ranked_lists": ranked_lists,
            "lexical_hit_count": lexical_hit_count,
            "vector_hit_count": vector_hit_count,
            "task_traces": task_traces,
            "parallel_workers": max_workers,
        }

    def _retrieve_route(
        self,
        rewritten_query: str,
        source_type: str,
        limit: int,
    ) -> Dict[str, Any]:
        with self.retriever._connect() as conn:
            lexical_hits = self._retrieve_from_source(conn, rewritten_query, source_type, limit)
        semantic_hits = self.retriever.vector_store.search(rewritten_query, source_type, limit)
        return {
            "rewritten_query": rewritten_query,
            "source_type": source_type,
            "lexical_hits": lexical_hits,
            "semantic_hits": semantic_hits,
        }

    def _retrieve_from_source(
        self,
        conn: Any,
        query: str,
        source_type: str,
        limit: int,
    ) -> List[IndexedChunk]:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE source_type = ?",
            (source_type,),
        ).fetchall()
        if not rows:
            return []

        contents = [row["content"] for row in rows]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(contents + [query])
        semantic_scores = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
        keyword_scores = bm25_scores(query, contents)
        merged: List[IndexedChunk] = []
        for idx, row in enumerate(rows):
            merged.append(
                IndexedChunk(
                    chunk_id=row["id"],
                    document_id=row["document_id"],
                    source_type=source_type,
                    content=row["content"],
                    metadata=json.loads(row["metadata"]),
                    score=settings.rag_cc_alpha * float(keyword_scores[idx]) + (1 - settings.rag_cc_alpha) * float(semantic_scores[idx]),
                )
            )
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:limit]
