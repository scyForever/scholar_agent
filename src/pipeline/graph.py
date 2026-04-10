from __future__ import annotations

from typing import Any, Dict, List

from src.core.models import ExecutionMode, SearchResult
from src.harness.contracts import MultiAgentHarnessRequest
from src.harness.multi_agent_harness import MultiAgentHarness


class MultiAgentPipeline:
    """兼容旧接口的 pipeline 包装，内部统一委托给 harness。"""

    def __init__(self, coordinator: Any) -> None:
        self.coordinator = coordinator
        self.harness = MultiAgentHarness(coordinator)
        self.graph = self.harness.graph

    def uses_langgraph(self) -> bool:
        return self.harness.uses_langgraph()

    def execute(
        self,
        *,
        query: str,
        intent: str,
        slots: Dict[str, Any],
        mode: ExecutionMode,
        trace_id: str,
        flow: List[str],
        task_config: Any = None,
        history: List[Dict[str, str]] | None = None,
        session_id: str = "",
        prior_search_result: SearchResult | None = None,
    ) -> Dict[str, Any]:
        return self.harness.execute(
            MultiAgentHarnessRequest(
                query=query,
                intent=intent,
                slots=slots,
                mode=mode,
                trace_id=trace_id,
                flow=flow,
                task_config=task_config,
                history=history or [],
                session_id=session_id,
                prior_search_result=prior_search_result,
            )
        )
