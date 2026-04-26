from __future__ import annotations

import hashlib
import json
import math
import re
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


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "用户",
    "系统",
    "回答",
    "问题",
    "请",
    "需要",
    "一个",
    "以及",
    "进行",
    "关于",
    "论文",
}
CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+\-.]{1,}|\d{4}")


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().replace("/", " ").split() if token]


def _owner_key(user_id: str) -> str:
    normalized = (user_id or "default").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _metadata_text(metadata: Dict[str, Any]) -> str:
    values: list[str] = []
    for value in metadata.values():
        if isinstance(value, (str, int, float)):
            values.append(str(value))
        elif isinstance(value, list):
            values.extend(str(item) for item in value if isinstance(item, (str, int, float)))
    return " ".join(values)


def _extract_keywords(text: str, *, limit: int = 18) -> List[str]:
    counter: Counter[str] = Counter()
    lowered = text.lower()
    for token in WORD_RE.findall(lowered):
        if token in STOPWORDS or len(token) < 2:
            continue
        counter[token] += 2

    for segment in CJK_RE.findall(text):
        if 2 <= len(segment) <= 8 and segment not in STOPWORDS:
            counter[segment] += 3
        max_ngram = min(6, len(segment))
        for size in range(2, max_ngram + 1):
            for index in range(0, len(segment) - size + 1):
                token = segment[index : index + size]
                if token in STOPWORDS:
                    continue
                counter[token] += 1 + size / 10

    return [item for item, _ in counter.most_common(limit)]


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


def _tfidf_scores(query: str, contents: List[str]) -> List[float]:
    if not contents:
        return []
    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(contents + [query])
        return [float(item) for item in cosine_similarity(matrix[:-1], matrix[-1]).ravel()]
    except ValueError:
        return [0.0 for _ in contents]


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
                    owner_key TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_owner_type ON memories (owner_key, type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_owner_updated ON memories (owner_key, updated_at)")
            conn.commit()

    def _load_metadata(self, raw: str | None) -> Dict[str, Any]:
        try:
            loaded = json.loads(raw or "{}")
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _row_keywords(self, raw: str | None) -> List[str]:
        try:
            loaded = json.loads(raw or "[]")
            if isinstance(loaded, list):
                return [str(item).strip().lower() for item in loaded if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return []

    def _user_profile_keywords(self, conn: sqlite3.Connection, user_id: str) -> set[str]:
        rows = conn.execute(
            """
            SELECT keywords, metadata
            FROM memories
            WHERE owner_key = ? AND type IN (?, ?)
            """,
            (
                _owner_key(user_id),
                MemoryType.PREFERENCE.value,
                MemoryType.RESEARCH_NOTE.value,
            ),
        ).fetchall()
        profile: set[str] = set()
        for row in rows:
            metadata = self._load_metadata(row["metadata"])
            profile.update(self._row_keywords(row["keywords"]))
        return profile

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
        metadata_payload = dict(metadata or {})
        keywords = _extract_keywords(f"{content}\n{_metadata_text(metadata_payload)}")
        metadata_payload.setdefault("keywords", keywords)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (id, user_id, owner_key, type, content, keywords, metadata, importance, access_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    user_id,
                    _owner_key(user_id),
                    memory_type.value,
                    content,
                    json.dumps(keywords, ensure_ascii=False),
                    json.dumps(metadata_payload, ensure_ascii=False),
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
        memory_types: Optional[List[MemoryType]] = None,
        user_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryRecord]:
        where: List[str] = []
        params: List[Any] = []
        selected_types = list(memory_types or [])
        if memory_type is not None:
            selected_types.append(memory_type)
        if selected_types:
            placeholders = ", ".join("?" for _ in selected_types)
            where.append(f"type IN ({placeholders})")
            params.extend(item.value for item in selected_types)
        if user_id is not None:
            where.append("owner_key = ?")
            params.append(_owner_key(user_id))
        sql = "SELECT * FROM memories"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                return []

            contents = [row["content"] for row in rows]
            similarities = _tfidf_scores(query, contents)
            bm25 = _bm25_scores(query, contents)
            now = datetime.utcnow()
            query_keywords = set(_extract_keywords(query, limit=12))
            profile_keywords = self._user_profile_keywords(conn, user_id) if user_id else set()

            scored: List[tuple[bool, MemoryRecord]] = []
            for idx, row in enumerate(rows):
                created_at = datetime.fromisoformat(row["created_at"])
                recency_days = max((now - created_at).days, 0)
                recency_bonus = math.exp(-recency_days / 30)
                metadata = self._load_metadata(row["metadata"])
                row_keywords = set(self._row_keywords(row["keywords"]))
                keyword_overlap = row_keywords.intersection(query_keywords)
                keyword_bonus = len(keyword_overlap) / max(len(query_keywords), 1) if query_keywords else 0.0
                profile_overlap = row_keywords.intersection(profile_keywords)
                profile_bonus = min(len(profile_overlap) / 5, 1.0) if profile_keywords else 0.0
                if query_keywords and query_keywords.intersection(profile_keywords):
                    profile_bonus = min(profile_bonus + 0.2, 1.0)
                score = (
                    0.35 * float(similarities[idx])
                    + 0.2 * float(bm25[idx])
                    + 0.2 * float(row["importance"])
                    + 0.1 * recency_bonus
                    + 0.1 * keyword_bonus
                    + 0.05 * profile_bonus
                )
                if query_keywords and not keyword_overlap:
                    score *= 0.75
                scored.append(
                    (
                        bool(keyword_overlap),
                        MemoryRecord(
                            memory_id=row["id"],
                            user_id=row["user_id"],
                            content=row["content"],
                            memory_type=MemoryType(row["type"]),
                            metadata=metadata,
                            importance=row["importance"],
                            access_count=row["access_count"],
                            created_at=created_at,
                            updated_at=datetime.fromisoformat(row["updated_at"]),
                            score=score,
                        ),
                    )
                )

            scored.sort(key=lambda item: item[1].score, reverse=True)
            if query_keywords and any(keyword_matched for keyword_matched, _ in scored):
                matched = [record for keyword_matched, record in scored if keyword_matched]
                unmatched = [record for keyword_matched, record in scored if not keyword_matched]
                selected = [*matched, *unmatched][:limit]
            else:
                selected = [record for _, record in scored[:limit]]
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
                    metadata=self._load_metadata(row["metadata"]),
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
                "paper_keywords": paper.keywords,
                "categories": paper.categories,
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
        allowed_types = {
            MemoryType.PREFERENCE,
            MemoryType.PAPER_SUMMARY,
            MemoryType.RESEARCH_NOTE,
            MemoryType.KNOWLEDGE,
        }
        return self.recall(
            query,
            user_id=user_id,
            memory_types=list(allowed_types),
            limit=limit,
        )

    def seen_paper_keys(self, user_id: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT metadata, content
                FROM memories
                WHERE owner_key = ? AND type IN (?, ?, ?)
                """,
                (
                    _owner_key(user_id),
                    MemoryType.PAPER_SUMMARY.value,
                    MemoryType.RESEARCH_NOTE.value,
                    MemoryType.KNOWLEDGE.value,
                ),
            ).fetchall()
        keys: set[str] = set()
        for row in rows:
            metadata = self._load_metadata(row["metadata"])
            for field in ("paper_id", "title", "doi", "arxiv_id", "pmid"):
                value = str(metadata.get(field) or "").strip().lower()
                if value:
                    keys.add(value)
            content = str(row["content"] or "").strip().lower()
            if content:
                keys.add(content)
        return keys

    def format_recall_context(self, records: List[MemoryRecord]) -> str:
        if not records:
            return ""
        lines = ["长期记忆-用户专属召回："]
        for record in records:
            keywords = record.metadata.get("keywords") or []
            keyword_text = "、".join(str(item) for item in keywords[:5]) if isinstance(keywords, list) else ""
            prefix = f"- [{record.memory_type.value} score={record.score:.3f}]"
            if keyword_text:
                prefix += f" 关键词：{keyword_text}"
            lines.append(f"{prefix}\n{record.content}")
        return "\n".join(lines)
