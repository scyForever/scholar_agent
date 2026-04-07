from __future__ import annotations

from typing import Any

__all__ = [
    "AgentRuntimeGraph",
    "MultiAgentPipeline",
    "MultiAgentState",
    "RuntimeState",
]


def __getattr__(name: str) -> Any:
    if name == "AgentRuntimeGraph":
        from .runtime_graph import AgentRuntimeGraph

        return AgentRuntimeGraph
    if name == "MultiAgentPipeline":
        from .graph import MultiAgentPipeline

        return MultiAgentPipeline
    if name in {"MultiAgentState", "RuntimeState"}:
        from .state import MultiAgentState, RuntimeState

        mapping = {
            "MultiAgentState": MultiAgentState,
            "RuntimeState": RuntimeState,
        }
        return mapping[name]
    raise AttributeError(name)
