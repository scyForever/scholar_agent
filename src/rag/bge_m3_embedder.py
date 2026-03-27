from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from config.settings import settings


class BGEM3Embedder:
    def __init__(
        self,
        model_path: str | None = None,
        *,
        batch_size: int | None = None,
        max_length: int | None = None,
        use_fp16: bool | None = None,
    ) -> None:
        self.model_path = Path(model_path or settings.bge_m3_model_path).expanduser() if (model_path or settings.bge_m3_model_path) else None
        self.batch_size = batch_size or settings.bge_m3_batch_size
        self.max_length = max_length or settings.bge_m3_max_length
        self.use_fp16 = settings.bge_m3_use_fp16 if use_fp16 is None else use_fp16
        self._model: Any | None = None

    def status(self) -> Tuple[bool, str]:
        if self.model_path is None:
            return False, "BGE_M3_MODEL_PATH is not configured."
        if not self.model_path.exists():
            return False, f"BGE-M3 model path does not exist: {self.model_path}"
        try:
            from FlagEmbedding import BGEM3FlagModel  # noqa: F401
        except Exception as exc:
            return False, f"FlagEmbedding import failed: {type(exc).__name__}: {exc}"
        return True, ""

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._load_model()
        encoded = model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        dense_vecs = encoded["dense_vecs"]
        if hasattr(dense_vecs, "tolist"):
            dense_vecs = dense_vecs.tolist()
        return [
            [float(value) for value in vector]
            for vector in dense_vecs
        ]

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

    def _load_model(self) -> Any:
        ok, reason = self.status()
        if not ok:
            raise RuntimeError(reason)
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(str(self.model_path), use_fp16=self.use_fp16)
        return self._model
