from __future__ import annotations

import json
import math
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from uuid import uuid4

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from src.core.llm import LLMManager
from src.core.models import IndexedChunk
from src.preprocessing.query_rewriter import QueryRewritePlan, QueryRewriter
from src.rag.bge_m3_embedder import BGEM3Embedder
from src.rag.bge_reranker import BGEReranker
from src.rag.vector_store import LocalChromaVectorStore, VectorChunkRecord
from src.tools.pdf_tool import extract_pdf_text
from src.tools.web_search_tool import search_web


def _tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"[\s,.;:!?()，。；：！？、]+", text.lower()) if token]


def _bm25_scores(query: str, documents: Iterable[str]) -> List[float]:
    docs = list(documents)
    tokenized_docs = [_tokenize(doc) for doc in docs]
    query_tokens = _tokenize(query)
    if not docs or not query_tokens:
        return [0.0 for _ in docs]

    doc_count = len(tokenized_docs)
    avgdl = sum(len(doc) for doc in tokenized_docs) / max(doc_count, 1)
    doc_freq: Counter[str] = Counter()
    for doc in tokenized_docs:
        for token in set(doc):
            doc_freq[token] += 1

    scores: List[float] = []
    k1 = 1.5
    b = 0.75
    for doc in tokenized_docs:
        tf = Counter(doc)
        doc_len = len(doc) or 1
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
            numerator = tf[token] * (k1 + 1)
            denominator = tf[token] + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * numerator / denominator
        scores.append(score)
    return scores


class HybridRetriever:
    def __init__(self, db_path: Path | None = None, llm: LLMManager | None = None) -> None:
        self.db_path = db_path or (settings.memory_dir / "rag_index.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm or LLMManager()
        self.rewriter = QueryRewriter(self.llm)
        self.embedder = BGEM3Embedder()
        self.reranker = BGEReranker()
        self.vector_store = LocalChromaVectorStore(embedder=self.embedder)
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
            conn.commit()

    def index_pdf(self, pdf_path: str, title: str | None = None, metadata: Dict[str, Any] | None = None) -> str:
        ok, reason = self.vector_store.status()
        if not ok:
            raise RuntimeError(reason)
        extracted = extract_pdf_text(pdf_path)
        document_id = str(uuid4())
        document_title = title or Path(pdf_path).stem
        merged_metadata = {"pdf_path": pdf_path, **(metadata or {})}
        base_chunk_metadata = {"title": document_title, **merged_metadata}
        records = [
            *self._build_chunk_records(document_id, extracted["chunks"], "text_chunk", base_chunk_metadata),
            *self._build_chunk_records(document_id, extracted["tables"], "table_chunk", base_chunk_metadata),
            *self._build_chunk_records(document_id, [item["answer"] for item in extracted["qa_pairs"]], "qa_chunk", base_chunk_metadata),
            *self._build_chunk_records(document_id, self._build_kg_triples(extracted["chunks"]), "kg_chunk", base_chunk_metadata),
        ]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, title, metadata, created_at) VALUES (?, ?, ?, ?)",
                (
                    document_id,
                    document_title,
                    json.dumps(merged_metadata, ensure_ascii=False),
                    datetime.utcnow().isoformat(),
                ),
            )
            self._store_chunk_records(conn, records)
            conn.commit()
        try:
            self.vector_store.upsert(records)
        except Exception:
            with self._connect() as conn:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                conn.commit()
            raise
        return document_id

    def index_text(self, title: str, text: str, metadata: Dict[str, Any] | None = None) -> str:
        ok, reason = self.vector_store.status()
        if not ok:
            raise RuntimeError(reason)
        document_id = str(uuid4())
        base_chunk_metadata = {"title": title, **(metadata or {})}
        chunks = self._chunk_text(text)
        records = [
            *self._build_chunk_records(document_id, chunks, "text_chunk", base_chunk_metadata),
            *self._build_chunk_records(document_id, self._build_kg_triples(chunks), "kg_chunk", base_chunk_metadata),
        ]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, title, metadata, created_at) VALUES (?, ?, ?, ?)",
                (
                    document_id,
                    title,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    datetime.utcnow().isoformat(),
                ),
            )
            self._store_chunk_records(conn, records)
            conn.commit()
        try:
            self.vector_store.upsert(records)
        except Exception:
            with self._connect() as conn:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                conn.commit()
            raise
        return document_id

    def retrieve(
        self,
        query: str,
        chat_history: List[Dict[str, str]] | None = None,
        top_k: int | None = None,
        rewritten_queries: List[str] | None = None,
        rewrite_plan: QueryRewritePlan | None = None,
    ) -> Dict[str, Any]:
        top_k = top_k or settings.rag_top_k
        enhanced_query = self._conversation_enhance(query, chat_history or [])
        rewrites = list(rewritten_queries or self._rewrite_queries(enhanced_query))[:8]
        routes = self._route_sources(enhanced_query)
        vector_enabled, vector_reason = self.vector_store.status()
        reranker_enabled, reranker_reason = self.reranker.status()
        indexed_chunk_count = self._indexed_chunk_count()

        if indexed_chunk_count == 0:
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
        if not vector_enabled:
            raise RuntimeError(f"Local RAG vector index is unavailable: {vector_reason}")
        if not reranker_enabled:
            raise RuntimeError(f"BGE reranker is unavailable: {reranker_reason}")
        vector_chunk_count = self.vector_store.count()
        if vector_chunk_count < indexed_chunk_count:
            raise RuntimeError(
                "Local RAG vector index is incomplete. Re-index local documents after configuring BGE-M3."
            )

        ranked_lists: List[List[IndexedChunk]] = []
        lexical_hit_count = 0
        vector_hit_count = 0
        task_traces: List[Dict[str, Any]] = []
        tasks = [(rewritten_query, source_type) for rewritten_query in rewrites for source_type in routes]
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

        fused = self._rrf_fusion(ranked_lists)
        reranked = self._rerank(enhanced_query, fused)
        validated, supplement = self._crag_validate(enhanced_query, reranked[:top_k])

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
                "parallel_workers": max_workers,
                "parallel_tasks": sorted(
                    task_traces,
                    key=lambda item: (str(item["rewritten_query"]), str(item["source_type"])),
                ),
                "lexical_hit_count": lexical_hit_count,
                "vector_hit_count": vector_hit_count,
                "before_fusion": sum(len(items) for items in ranked_lists),
                "after_rerank": len(reranked),
                "validated_count": len(validated),
            },
        }

    def _build_chunk_records(
        self,
        document_id: str,
        items: Iterable[str],
        source_type: str,
        metadata: Dict[str, Any] | None = None,
    ) -> List[VectorChunkRecord]:
        records: List[VectorChunkRecord] = []
        for content in items:
            content = str(content).strip()
            if not content:
                continue
            records.append(
                VectorChunkRecord(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    source_type=source_type,
                    content=content,
                    metadata=dict(metadata or {}),
                )
            )
        return records

    def _store_chunk_records(
        self,
        conn: sqlite3.Connection,
        records: Iterable[VectorChunkRecord],
    ) -> None:
        for record in records:
            conn.execute(
                "INSERT INTO chunks (id, document_id, source_type, content, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    record.chunk_id,
                    record.document_id,
                    record.source_type,
                    record.content,
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )

    def _chunk_text(self, text: str) -> List[str]:
        text = " ".join(text.split())
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + settings.rag_chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = max(end - settings.rag_chunk_overlap, start + 1)
        return chunks

    def _build_kg_triples(self, chunks: Iterable[str]) -> List[str]:
        triples: List[str] = []
        for chunk in chunks:
            entities = [token for token in _tokenize(chunk) if len(token) > 4]
            unique_entities = list(dict.fromkeys(entities[:6]))
            for idx in range(len(unique_entities) - 1):
                triples.append(f"{unique_entities[idx]} related_to {unique_entities[idx + 1]}")
        return triples

    def _conversation_enhance(self, query: str, history: List[Dict[str, str]]) -> str:
        if not history:
            return query
        recent = " ".join(item["content"] for item in history[-4:] if item["role"] == "user")
        if recent and query not in recent:
            return f"{recent} {query}"
        return query

    def _rewrite_queries(self, query: str) -> List[str]:
        return self.rewriter.rewrite(query, intent="search_papers", target="local")[:8]

    def _indexed_chunk_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"] if row is not None else 0)

    def _route_sources(self, query: str) -> List[str]:
        routes = ["text_chunk"]
        lowered = query.lower()
        if any(token in lowered for token in ("表", "table", "benchmark", "指标")):
            routes.append("table_chunk")
        if any(token in lowered for token in ("why", "how", "是什么", "如何", "为什么")):
            routes.append("qa_chunk")
        if any(token in lowered for token in ("关系", "演化", "关联", "graph")):
            routes.append("kg_chunk")
        return list(dict.fromkeys(routes))

    def _retrieve_route(
        self,
        rewritten_query: str,
        source_type: str,
        limit: int,
    ) -> Dict[str, Any]:
        with self._connect() as conn:
            lexical_hits = self._retrieve_from_source(conn, rewritten_query, source_type, limit)
        semantic_hits = self.vector_store.search(rewritten_query, source_type, limit)
        return {
            "rewritten_query": rewritten_query,
            "source_type": source_type,
            "lexical_hits": lexical_hits,
            "semantic_hits": semantic_hits,
        }

    def _retrieve_from_source(
        self,
        conn: sqlite3.Connection,
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
        keyword_scores = _bm25_scores(query, contents)
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

    def _rrf_fusion(self, ranked_lists: List[List[IndexedChunk]]) -> List[IndexedChunk]:
        scores: Dict[str, float] = defaultdict(float)
        best_items: Dict[str, IndexedChunk] = {}
        for items in ranked_lists:
            for rank, item in enumerate(items, start=1):
                scores[item.chunk_id] += 1.0 / (settings.rag_rrf_k + rank)
                if item.chunk_id not in best_items or item.score > best_items[item.chunk_id].score:
                    best_items[item.chunk_id] = item

        fused = []
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

    def _rerank(self, query: str, chunks: List[IndexedChunk]) -> List[IndexedChunk]:
        return self.reranker.rerank(query, chunks)

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
        result = self.llm.call_json(prompt, purpose="RAG相关性判断")
        label = str(result.get("label", "")).lower()
        if label in {"correct", "ambiguous"}:
            return True
        if label == "incorrect":
            return False
        return score >= 0.18
