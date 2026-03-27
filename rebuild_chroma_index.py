from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from config.settings import settings
from src.rag.vector_store import LocalChromaVectorStore, VectorChunkRecord


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the local ChromaDB index from SQLite chunks.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=settings.memory_dir / "rag_index.db",
        help="Path to the SQLite chunk database.",
    )
    parser.add_argument(
        "--collection",
        default=settings.vector_collection_name,
        help="ChromaDB collection directory name under data/vector_db.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of chunks to embed and write per batch.",
    )
    parser.add_argument(
        "--source-type",
        action="append",
        dest="source_types",
        help="Optional source_type filter. Can be passed multiple times.",
    )
    return parser.parse_args()


def _batched(items: Iterable[VectorChunkRecord], batch_size: int) -> Iterator[List[VectorChunkRecord]]:
    batch: List[VectorChunkRecord] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_json(value: Any) -> Dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _count_chunks(db_path: Path, source_types: List[str] | None) -> int:
    with sqlite3.connect(db_path) as conn:
        if source_types:
            placeholders = ", ".join("?" for _ in source_types)
            row = conn.execute(
                f"SELECT COUNT(*) FROM chunks WHERE source_type IN ({placeholders})",
                tuple(source_types),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    return int(row[0] if row is not None else 0)


def _iter_chunk_records(db_path: Path, source_types: List[str] | None) -> Iterator[VectorChunkRecord]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        sql = (
            "SELECT "
            "c.id AS chunk_id, "
            "c.document_id AS document_id, "
            "c.source_type AS source_type, "
            "c.content AS content, "
            "c.metadata AS chunk_metadata, "
            "d.title AS document_title, "
            "d.metadata AS document_metadata "
            "FROM chunks c "
            "LEFT JOIN documents d ON d.id = c.document_id "
        )
        params: List[str] = []
        if source_types:
            placeholders = ", ".join("?" for _ in source_types)
            sql += f"WHERE c.source_type IN ({placeholders}) "
            params.extend(source_types)
        sql += "ORDER BY c.document_id, c.source_type, c.id"

        cursor = conn.execute(sql, params)
        for row in cursor:
            document_metadata = _load_json(row["document_metadata"])
            chunk_metadata = _load_json(row["chunk_metadata"])
            metadata: Dict[str, Any] = dict(document_metadata)
            title = row["document_title"]
            if isinstance(title, str) and title:
                metadata.setdefault("title", title)
            metadata.update(chunk_metadata)

            content = str(row["content"] or "").strip()
            if not content:
                continue

            yield VectorChunkRecord(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                source_type=str(row["source_type"]),
                content=content,
                metadata=metadata,
            )
    finally:
        conn.close()


def main() -> int:
    args = _parse_args()
    db_path = args.db_path.expanduser().resolve()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    total_chunks = _count_chunks(db_path, args.source_types)
    if total_chunks == 0:
        print(f"No chunks found in {db_path}. Nothing to rebuild.")
        return 0

    vector_store = LocalChromaVectorStore(collection_name=args.collection)
    ok, reason = vector_store.status()
    if not ok:
        raise RuntimeError(reason)

    vector_store.clear()

    rebuilt = 0
    for batch in _batched(_iter_chunk_records(db_path, args.source_types), args.batch_size):
        vector_store.upsert(batch)
        rebuilt += len(batch)
        print(f"Indexed {rebuilt}/{total_chunks} chunks into ChromaDB collection '{args.collection}'.")

    final_count = vector_store.count()
    if final_count != rebuilt:
        raise RuntimeError(f"ChromaDB rebuild count mismatch: expected {rebuilt}, got {final_count}.")

    print(f"ChromaDB rebuild complete. Collection path: {vector_store.storage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
