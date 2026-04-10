from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from src.core.models import ExecutionMode, SearchResult
from src.planning.task_hierarchy import TaskConfig


@dataclass(slots=True)
class MultiAgentHarnessRequest:
    query: str
    intent: str
    slots: Dict[str, Any]
    mode: ExecutionMode
    trace_id: str
    flow: List[str]
    task_config: TaskConfig | None = None
    history: List[Dict[str, str]] = field(default_factory=list)
    session_id: str = ""
    prior_search_result: SearchResult | None = None


@dataclass(slots=True)
class RuntimeHarnessRequest:
    query: str
    intent: str
    slots: Dict[str, Any]
    session_id: str
    trace_id: str
    task_config: TaskConfig
    history: List[Dict[str, str]] = field(default_factory=list)
    memory_context: str = ""
    prior_search_result: SearchResult | None = None
    execution_mode: ExecutionMode = ExecutionMode.STANDARD
    enable_quality_enhance: bool = False


@dataclass(slots=True)
class ScholarChatRequest:
    query: str
    session_id: str = "default"
    on_trace_start: Callable[[str], None] | None = None
