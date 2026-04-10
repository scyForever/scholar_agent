from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List
from uuid import uuid4

from config.settings import settings
from src.rag.components.common import tokenize
from src.rag.vector_store import VectorChunkRecord
from src.tools.pdf_tool import extract_pdf_text


class RAGIndexingComponent:
    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def index_pdf(self, pdf_path: str, title: str | None = None, metadata: Dict[str, Any] | None = None) -> str:
        ok, reason = self.retriever.vector_store.status()
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
        with self.retriever._connect() as conn:
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
            self.retriever.vector_store.upsert(records)
        except Exception:
            with self.retriever._connect() as conn:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                conn.commit()
            raise
        return document_id

    def index_text(self, title: str, text: str, metadata: Dict[str, Any] | None = None) -> str:
        ok, reason = self.retriever.vector_store.status()
        if not ok:
            raise RuntimeError(reason)
        document_id = str(uuid4())
        base_chunk_metadata = {"title": title, **(metadata or {})}
        chunks = self._chunk_text(text)
        records = [
            *self._build_chunk_records(document_id, chunks, "text_chunk", base_chunk_metadata),
            *self._build_chunk_records(document_id, self._build_kg_triples(chunks), "kg_chunk", base_chunk_metadata),
        ]
        with self.retriever._connect() as conn:
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
            self.retriever.vector_store.upsert(records)
        except Exception:
            with self.retriever._connect() as conn:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                conn.commit()
            raise
        return document_id

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
        conn: Any,
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
            entities = [token for token in tokenize(chunk) if len(token) > 4]
            unique_entities = list(dict.fromkeys(entities[:6]))
            for idx in range(len(unique_entities) - 1):
                triples.append(f"{unique_entities[idx]} related_to {unique_entities[idx + 1]}")
        return triples
