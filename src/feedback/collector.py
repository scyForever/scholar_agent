from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config.settings import settings


class FeedbackCollector:
    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or settings.feedback_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.touch(exist_ok=True)

    def record_feedback(
        self,
        session_id: str,
        query: str,
        response: str,
        rating: int,
        comment: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "session_id": session_id,
            "query": query,
            "response": response,
            "rating": rating,
            "comment": comment,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        with self.storage_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_feedback(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with self.storage_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def summarize(self) -> Dict[str, Any]:
        records = self.load_feedback()
        if not records:
            return {"count": 0, "average_rating": 0.0, "common_comments": []}
        ratings = [item["rating"] for item in records]
        comments = Counter(
            item["comment"] for item in records if item.get("comment")
        ).most_common(5)
        return {
            "count": len(records),
            "average_rating": sum(ratings) / len(ratings),
            "common_comments": comments,
        }
