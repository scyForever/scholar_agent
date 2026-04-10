from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from config.settings import settings
from src.core.llm import LLMManager
from src.preprocessing.query_rewriter import QueryRewritePlan, QueryRewriter
from src.rag.bge_m3_embedder import BGEM3Embedder
from src.rag.bge_reranker import BGEReranker
from src.rag.components import RAGIndexingComponent, RAGRetrievalComponent
from src.rag.contracts import RAGIndexPDFRequest, RAGIndexTextRequest, RAGRetrieveRequest
from src.rag.harness import RAGHarness
from src.rag.vector_store import LocalChromaVectorStore


class HybridRetriever:
    """兼容旧接口的 RAG 入口，底层索引与检索已拆分为独立组件。"""

    def __init__(self, db_path: Path | None = None, llm: LLMManager | None = None) -> None:
        self.db_path = db_path or (settings.memory_dir / "rag_index.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm or LLMManager()
        self.rewriter = QueryRewriter(self.llm)
        self.embedder = BGEM3Embedder()
        self.reranker = BGEReranker()
        self.vector_store = LocalChromaVectorStore(embedder=self.embedder)
        self._create_tables()
        self.indexing_component = RAGIndexingComponent(self)
        self.retrieval_component = RAGRetrievalComponent(self)
        self.harness = RAGHarness(self)

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
        return self.harness.index_pdf(
            RAGIndexPDFRequest(
                pdf_path=pdf_path,
                title=title,
                metadata=metadata,
            )
        )

    def _index_pdf_impl(self, pdf_path: str, title: str | None = None, metadata: Dict[str, Any] | None = None) -> str:
        return self.indexing_component.index_pdf(pdf_path, title=title, metadata=metadata)

    def index_text(self, title: str, text: str, metadata: Dict[str, Any] | None = None) -> str:
        return self.harness.index_text(
            RAGIndexTextRequest(
                title=title,
                text=text,
                metadata=metadata,
            )
        )

    def _index_text_impl(self, title: str, text: str, metadata: Dict[str, Any] | None = None) -> str:
        return self.indexing_component.index_text(title, text, metadata=metadata)

    def retrieve(
        self,
        query: str,
        chat_history: List[Dict[str, str]] | None = None,
        top_k: int | None = None,
        rewritten_queries: List[str] | None = None,
        rewrite_plan: QueryRewritePlan | None = None,
    ) -> Dict[str, Any]:
        return self.harness.retrieve(
            RAGRetrieveRequest(
                query=query,
                chat_history=chat_history or [],
                top_k=top_k,
                rewritten_queries=rewritten_queries,
                rewrite_plan=rewrite_plan,
            )
        )

    def _retrieve_impl(
        self,
        query: str,
        chat_history: List[Dict[str, str]] | None = None,
        top_k: int | None = None,
        rewritten_queries: List[str] | None = None,
        rewrite_plan: QueryRewritePlan | None = None,
    ) -> Dict[str, Any]:
        return self.retrieval_component.retrieve(
            query,
            chat_history=chat_history,
            top_k=top_k,
            rewritten_queries=rewritten_queries,
            rewrite_plan=rewrite_plan,
        )
