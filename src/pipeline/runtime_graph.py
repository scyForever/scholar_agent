from __future__ import annotations

from typing import Any, Dict, List

from src.core.models import ExecutionMode, SearchResult
from src.harness.contracts import RuntimeHarnessRequest
from src.harness.runtime_harness import RuntimeHarness
from src.planning.task_hierarchy import TaskConfig


class AgentRuntimeGraph:
    """兼容旧接口的 runtime 包装，内部统一委托给 harness。"""

    def __init__(
        self,
        multi_agent: Any,
        reasoning: Any,
        quality: Any,
        tracer: Any,
    ) -> None:
        self.multi_agent = multi_agent
        self.reasoning = reasoning
        self.quality = quality
        self.tracer = tracer
        self.harness = RuntimeHarness(
            multi_agent=multi_agent,
            reasoning=reasoning,
            quality=quality,
            tracer=tracer,
        )
        self.graph = self.harness.graph

    def uses_langgraph(self) -> bool:
        return self.harness.uses_langgraph()

    def execute(
        self,
        *,
        query: str,
        intent: str,
        slots: Dict[str, Any],
        session_id: str,
        trace_id: str,
        task_config: TaskConfig,
        history: List[Dict[str, str]],
        memory_context: str,
        prior_search_result: SearchResult | None,
        execution_mode: ExecutionMode,
        enable_quality_enhance: bool,
    ) -> Dict[str, Any]:
        return self.harness.execute(
            RuntimeHarnessRequest(
                query=query,
                intent=intent,
                slots=slots,
                session_id=session_id,
                trace_id=trace_id,
                task_config=task_config,
                history=history,
                memory_context=memory_context,
                prior_search_result=prior_search_result,
                execution_mode=execution_mode,
                enable_quality_enhance=enable_quality_enhance,
            )
        )
