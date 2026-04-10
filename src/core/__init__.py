from __future__ import annotations

from typing import Any

__all__ = [
    "Agent",
    "AgentV2",
    "ScholarAgentHarness",
    "LLMManager",
    "Paper",
    "SearchResult",
    "AgentResponse",
    "DialogueState",
    "ExecutionMode",
]


def __getattr__(name: str) -> Any:
    if name == "Agent":
        from .agent import Agent

        return Agent
    if name == "AgentV2":
        from .agent_v2 import AgentV2

        return AgentV2
    if name == "ScholarAgentHarness":
        from src.harness import ScholarAgentHarness

        return ScholarAgentHarness
    if name == "LLMManager":
        from .llm import LLMManager

        return LLMManager
    if name in {"Paper", "SearchResult", "AgentResponse", "DialogueState", "ExecutionMode"}:
        from .models import AgentResponse, DialogueState, ExecutionMode, Paper, SearchResult

        mapping = {
            "Paper": Paper,
            "SearchResult": SearchResult,
            "AgentResponse": AgentResponse,
            "DialogueState": DialogueState,
            "ExecutionMode": ExecutionMode,
        }
        return mapping[name]
    raise AttributeError(name)
