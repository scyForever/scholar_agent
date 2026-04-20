from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from src.core.models import MemoryRecord, MemoryType, Paper


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().replace("/", " ").split() if token]


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


class MemoryManager:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.memory_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def store(
        self,
        user_id: str,
        content: str,
        *,
        memory_type: MemoryType = MemoryType.KNOWLEDGE,
        metadata: Dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> str:
        now = datetime.utcnow().isoformat()
        memory_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (id, user_id, type, content, metadata, importance, access_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    user_id,
                    memory_type.value,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    float(importance),
                    0,
                    now,
                    now,
                ),
            )
            conn.commit()
        return memory_id

    def recall(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        *,
        user_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryRecord]:
        where: List[str] = []
        params: List[Any] = []
        if memory_type is not None:
            where.append("type = ?")
            params.append(memory_type.value)
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        sql = "SELECT * FROM memories"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                return []

            contents = [row["content"] for row in rows]
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(contents + [query])
            similarities = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
            bm25 = _bm25_scores(query, contents)
            now = datetime.utcnow()

            scored: List[MemoryRecord] = []
            for idx, row in enumerate(rows):
                created_at = datetime.fromisoformat(row["created_at"])
                recency_days = max((now - created_at).days, 0)
                recency_bonus = math.exp(-recency_days / 30)
                score = (
                    0.45 * float(similarities[idx])
                    + 0.25 * float(bm25[idx])
                    + 0.2 * float(row["importance"])
                    + 0.1 * recency_bonus
                )
                scored.append(
                    MemoryRecord(
                        memory_id=row["id"],
                        user_id=row["user_id"],
                        content=row["content"],
                        memory_type=MemoryType(row["type"]),
                        metadata=json.loads(row["metadata"]),
                        importance=row["importance"],
                        access_count=row["access_count"],
                        created_at=created_at,
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                        score=score,
                    )
                )

            scored.sort(key=lambda item: item.score, reverse=True)
            selected = scored[:limit]
            for record in selected:
                conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, updated_at = ? WHERE id = ?",
                    (datetime.utcnow().isoformat(), record.memory_id),
                )
            conn.commit()
            return selected

    def forget(self, importance_threshold: float = 0.3, older_than_days: int = 90) -> int:
        cutoff = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM memories
                WHERE importance < ? AND updated_at < ?
                """,
                (importance_threshold, cutoff),
            )
            conn.commit()
            return cursor.rowcount

    def list_recent(self, limit: int = 10) -> List[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                MemoryRecord(
                    memory_id=row["id"],
                    user_id=row["user_id"],
                    content=row["content"],
                    memory_type=MemoryType(row["type"]),
                    metadata=json.loads(row["metadata"]),
                    importance=row["importance"],
                    access_count=row["access_count"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
                for row in rows
            ]

    def remember_preference(
        self,
        user_id: str,
        preference: str,
        *,
        metadata: Dict[str, Any] | None = None,
        importance: float = 0.85,
    ) -> str:
        return self.store(
            user_id,
            preference,
            memory_type=MemoryType.PREFERENCE,
            metadata=metadata,
            importance=importance,
        )

    def remember_paper(
        self,
        user_id: str,
        paper: Paper,
        summary: str,
        *,
        highlights: List[str] | None = None,
        importance: float = 0.8,
    ) -> str:
        content_lines = [
            f"论文：{paper.title}",
            f"来源：{paper.source}",
            f"年份：{paper.year or '未知'}",
            f"摘要总结：{summary.strip()}",
        ]
        if highlights:
            content_lines.append("核心观点：" + "；".join(item.strip() for item in highlights if item.strip()))
        return self.store(
            user_id,
            "\n".join(content_lines),
            memory_type=MemoryType.PAPER_SUMMARY,
            metadata={
                "paper_id": paper.paper_id,
                "title": paper.title,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "pmid": paper.pmid,
                "source": paper.source,
                "year": paper.year,
            },
            importance=importance,
        )

    def recall_research_context(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 8,
    ) -> List[MemoryRecord]:
        records = self.recall(query, user_id=user_id, limit=max(limit * 2, limit))
        allowed_types = {
            MemoryType.PREFERENCE,
            MemoryType.PAPER_SUMMARY,
            MemoryType.RESEARCH_NOTE,
            MemoryType.KNOWLEDGE,
        }
        filtered = [record for record in records if record.memory_type in allowed_types]
        return filtered[:limit]

    def seen_paper_keys(self, user_id: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT metadata, content
                FROM memories
                WHERE user_id = ? AND type IN (?, ?, ?)
                """,
                (
                    user_id,
                    MemoryType.PAPER_SUMMARY.value,
                    MemoryType.RESEARCH_NOTE.value,
                    MemoryType.KNOWLEDGE.value,
                ),
            ).fetchall()
        keys: set[str] = set()
        for row in rows:
            metadata = json.loads(row["metadata"] or "{}")
            for field in ("paper_id", "title", "doi", "arxiv_id", "pmid"):
                value = str(metadata.get(field) or "").strip().lower()
                if value:
                    keys.add(value)
            content = str(row["content"] or "").strip().lower()
            if content:
                keys.add(content)
        return keys
