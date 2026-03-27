from __future__ import annotations

from typing import Any

__all__ = ["AgentV2"]


def __getattr__(name: str) -> Any:
    if name == "AgentV2":
        from src.core.agent_v2 import AgentV2

        return AgentV2
    raise AttributeError(name)
