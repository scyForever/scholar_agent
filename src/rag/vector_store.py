from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from config.settings import settings
from src.core.models import IndexedChunk
from src.rag.bge_m3_embedder import BGEM3Embedder


@dataclass(slots=True)
class VectorChunkRecord:
    chunk_id: str
    document_id: str
    source_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalChromaVectorStore:
    def __init__(
        self,
        storage_path: Path | None = None,
        collection_name: str | None = None,
        embedder: BGEM3Embedder | None = None,
    ) -> None:
        self.root_path = storage_path or settings.vector_db_dir
        self.collection_name = collection_name or settings.vector_collection_name
        self.storage_path = self.root_path / self.collection_name
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or BGEM3Embedder()
        self._client: Any | None = None
        self._collection: Any | None = None

    def status(self) -> Tuple[bool, str]:
        ok, reason = self.embedder.status()
        if not ok:
            return ok, reason
        try:
            import chromadb  # noqa: F401
        except ImportError:
            return False, "chromadb is not installed."
        return True, ""

    def upsert(self, records: Iterable[VectorChunkRecord]) -> None:
        items = [record for record in records if record.content.strip()]
        if not items:
            return
        collection = self._get_collection()
        embeddings = self.embedder.embed_documents([record.content for record in items])
        metadatas = [
            {
                "document_id": record.document_id,
                "source_type": record.source_type,
                "metadata_json": json.dumps(record.metadata, ensure_ascii=False),
            }
            for record in items
        ]
        collection.upsert(
            ids=[record.chunk_id for record in items],
            documents=[record.content for record in items],
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, source_type: str, limit: int) -> List[IndexedChunk]:
        if limit <= 0:
            return []
        ok, _ = self.status()
        if not ok:
            return []
        collection = self._get_collection()
        response = collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=limit,
            where={"source_type": source_type},
            include=["documents", "metadatas", "distances"],
        )
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: List[IndexedChunk] = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            record_metadata: Dict[str, Any] = {}
            if isinstance(metadata, dict):
                metadata_json = metadata.get("metadata_json")
                if isinstance(metadata_json, str) and metadata_json:
                    try:
                        record_metadata = json.loads(metadata_json)
                    except json.JSONDecodeError:
                        record_metadata = {"raw_metadata_json": metadata_json}
                document_id = str(metadata.get("document_id") or "")
                stored_source_type = str(metadata.get("source_type") or source_type)
            else:
                document_id = ""
                stored_source_type = source_type

            score = 1.0 / (1.0 + float(distance or 0.0))
            results.append(
                IndexedChunk(
                    chunk_id=str(chunk_id),
                    document_id=document_id,
                    source_type=stored_source_type,
                    content=str(content or ""),
                    metadata=record_metadata,
                    score=score,
                )
            )
        return results

    def count(self) -> int:
        ok, reason = self.status()
        if not ok:
            raise RuntimeError(reason)
        collection = self._get_collection()
        return int(collection.count())

    def clear(self) -> None:
        if self.storage_path.exists():
            shutil.rmtree(self.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None

    def _get_collection(self) -> Any:
        ok, reason = self.status()
        if not ok:
            raise RuntimeError(reason)
        if self._collection is not None:
            return self._collection

        import chromadb

        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.storage_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection
