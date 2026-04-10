from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from config.settings import settings
from src.core.models import IndexedChunk
from src.tools.web_search_tool import search_web


class RAGFusionValidationComponent:
    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def fuse(self, ranked_lists: List[List[IndexedChunk]]) -> List[IndexedChunk]:
        scores: Dict[str, float] = defaultdict(float)
        best_items: Dict[str, IndexedChunk] = {}
        for items in ranked_lists:
            for rank, item in enumerate(items, start=1):
                scores[item.chunk_id] += 1.0 / (settings.rag_rrf_k + rank)
                if item.chunk_id not in best_items or item.score > best_items[item.chunk_id].score:
                    best_items[item.chunk_id] = item

        fused: List[IndexedChunk] = []
        for chunk_id, score in scores.items():
            item = best_items[chunk_id]
            fused.append(
                IndexedChunk(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    source_type=item.source_type,
                    content=item.content,
                    metadata=item.metadata,
                    score=score,
                )
            )
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused

    def rerank_and_validate(
        self,
        query: str,
        chunks: List[IndexedChunk],
        *,
        top_k: int,
    ) -> Tuple[List[IndexedChunk], List[IndexedChunk], List[Dict[str, str]]]:
        reranked = self.retriever.reranker.rerank(query, chunks)
        validated, supplement = self._crag_validate(query, reranked[:top_k])
        return reranked, validated, supplement

    def _crag_validate(self, query: str, chunks: List[IndexedChunk]) -> Tuple[List[IndexedChunk], List[Dict[str, str]]]:
        validated: List[IndexedChunk] = []
        for chunk in chunks:
            if self._judge_relevance(query, chunk.content, chunk.score):
                validated.append(chunk)
        supplement: List[Dict[str, str]] = []
        if len(validated) < 3:
            supplement = search_web(query, max_results=3)
        return validated, supplement

    def _judge_relevance(self, query: str, content: str, score: float) -> bool:
        prompt = (
            "请判断文档片段是否与查询高度相关，输出JSON："
            '{"label":"correct|incorrect|ambiguous","reason":"..."}\n'
            f"查询：{query}\n片段：{content[:800]}"
        )
        result = self.retriever.llm.call_json(prompt, purpose="RAG相关性判断")
        label = str(result.get("label", "")).lower()
        if label in {"correct", "ambiguous"}:
            return True
        if label == "incorrect":
            return False
        return score >= 0.18
