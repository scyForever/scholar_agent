from __future__ import annotations

from src.core.agent_v2 import AgentV2


class Agent:
    """Backward-compatible wrapper around AgentV2."""

    def __init__(self) -> None:
        self._agent = AgentV2()

    def chat(self, query: str, session_id: str = "default"):
        return self._agent.chat(query, session_id=session_id)
