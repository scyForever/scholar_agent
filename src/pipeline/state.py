from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from src.core.models import (
    DebateResult,
    ExecutionMode,
    PaperAnalysis,
    ResearchPlan,
    SearchResult,
)
from src.planning.task_hierarchy import TaskConfig


class MultiAgentState(TypedDict, total=False):
    query: str
    intent: str
    slots: Dict[str, Any]
    mode: ExecutionMode
    session_id: str
    trace_id: str
    task_config: TaskConfig | None
    history: List[Dict[str, str]]
    prior_search_result: SearchResult | None
    flow: List[str]
    research_plan: ResearchPlan | None
    search_result: SearchResult | None
    analyses: List[PaperAnalysis]
    debate: DebateResult | None
    answer: str
    artifacts: Dict[str, Any]


class RuntimeState(TypedDict, total=False):
    query: str
    intent: str
    slots: Dict[str, Any]
    session_id: str
    trace_id: str
    task_config: TaskConfig | None
    history: List[Dict[str, str]]
    memory_context: str
    prior_search_result: SearchResult | None
    execution_mode: ExecutionMode
    enable_quality_enhance: bool
    answer: str
    artifacts: Dict[str, Any]
