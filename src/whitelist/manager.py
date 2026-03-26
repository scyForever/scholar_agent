from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from config.settings import settings


DEFAULT_WHITELIST: Dict[str, List[str]] = {
    "search_agent": [
        "search_arxiv",
        "search_openalex",
        "search_semantic_scholar",
        "search_web_of_science",
        "search_web",
    ],
    "analyze_agent": ["extract_pdf_text"],
    "debate_agent": [],
    "write_agent": [],
    "coder_agent": [],
}


class WhitelistManager:
    def __init__(self, whitelist_path: Path | None = None) -> None:
        self.whitelist_path = whitelist_path or settings.whitelist_path
        self.whitelist_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_default()

    def _ensure_default(self) -> None:
        if not self.whitelist_path.exists():
            self.whitelist_path.write_text(
                json.dumps(DEFAULT_WHITELIST, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load(self) -> Dict[str, List[str]]:
        self._ensure_default()
        return json.loads(self.whitelist_path.read_text(encoding="utf-8"))

    def allowed_tools(self, agent_name: str) -> List[str]:
        return self.load().get(agent_name, [])

    def is_allowed(self, agent_name: str, tool_name: str) -> bool:
        return tool_name in self.allowed_tools(agent_name)

    def set_allowed_tools(self, agent_name: str, tools: List[str]) -> None:
        payload = self.load()
        payload[agent_name] = tools
        self.whitelist_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
