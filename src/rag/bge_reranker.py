from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

from config.settings import settings
from src.core.models import IndexedChunk


class BGEReranker:
    def __init__(
        self,
        model_path: str | None = None,
        *,
        batch_size: int | None = None,
        use_fp16: bool | None = None,
    ) -> None:
        configured_path = model_path or settings.bge_reranker_model_path
        self.model_path = Path(configured_path).expanduser() if configured_path else None
        self.batch_size = batch_size or settings.bge_reranker_batch_size
        self.use_fp16 = settings.bge_reranker_use_fp16 if use_fp16 is None else use_fp16
        self._model: Any | None = None

    def status(self) -> Tuple[bool, str]:
        if self.model_path is None:
            return False, "BGE_RERANKER_MODEL_PATH is not configured."
        if not self.model_path.exists():
            return False, f"BGE reranker model path does not exist: {self.model_path}"
        try:
            from FlagEmbedding import FlagReranker  # noqa: F401
        except Exception as exc:
            return False, f"FlagEmbedding import failed: {type(exc).__name__}: {exc}"
        return True, ""

    def rerank(self, query: str, chunks: Sequence[IndexedChunk]) -> List[IndexedChunk]:
        if not chunks:
            return []
        model = self._load_model()
        reranked: List[IndexedChunk] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = list(chunks[start : start + self.batch_size])
            pairs = [[query, chunk.content] for chunk in batch]
            scores = model.compute_score(pairs)
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
            for chunk, score in zip(batch, scores):
                reranked.append(
                    IndexedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        source_type=chunk.source_type,
                        content=chunk.content,
                        metadata=chunk.metadata,
                        score=float(score),
                    )
                )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    def _load_model(self) -> Any:
        ok, reason = self.status()
        if not ok:
            raise RuntimeError(reason)
        if self._model is None:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(str(self.model_path), use_fp16=self.use_fp16)
        return self._model
