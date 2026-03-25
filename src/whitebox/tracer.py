from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from config.settings import settings


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return obj


class WhiteboxTracer:
    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir or settings.trace_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._traces: Dict[str, Dict[str, Any]] = {}

    def start_trace(
        self,
        session_id: str,
        query: str,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        trace_id = str(uuid4())
        self._traces[trace_id] = {
            "trace_id": trace_id,
            "session_id": session_id,
            "query": query,
            "status": "running",
            "metadata": metadata or {},
            "steps": [],
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": None,
            "final_output": None,
        }
        self._persist(trace_id)
        return trace_id

    def trace_step(
        self,
        trace_id: str,
        step_type: str,
        input_data: Any,
        output_data: Any,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        trace = self._traces.setdefault(trace_id, {"steps": []})
        trace.setdefault("steps", []).append(
            {
                "type": step_type,
                "input": _jsonable(input_data),
                "output": _jsonable(output_data),
                "metadata": _jsonable(metadata or {}),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self._persist(trace_id)

    def finish_trace(self, trace_id: str, final_output: Any) -> None:
        trace = self._traces[trace_id]
        trace["status"] = "completed"
        trace["finished_at"] = datetime.utcnow().isoformat()
        trace["final_output"] = _jsonable(final_output)
        self._persist(trace_id)

    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        if trace_id not in self._traces:
            path = self.storage_dir / f"{trace_id}.json"
            if path.exists():
                self._traces[trace_id] = json.loads(path.read_text(encoding="utf-8"))
        return self._traces.get(trace_id, {})

    def get_reasoning_chain(self, trace_id: str) -> list[Dict[str, Any]]:
        return self.get_trace(trace_id).get("steps", [])

    def _persist(self, trace_id: str) -> None:
        path = self.storage_dir / f"{trace_id}.json"
        path.write_text(
            json.dumps(self._traces[trace_id], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
