from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from src.core.models import DebateResult, PaperAnalysis, ResearchPlan, SearchResult
from src.harness.base import BaseHarness
from src.harness.contracts import MultiAgentHarnessRequest
from src.pipeline.state import MultiAgentState

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover
    END = None
    START = None
    StateGraph = None

if TYPE_CHECKING:
    from src.agents.multi_agent import MultiAgentCoordinator


class MultiAgentHarness(BaseHarness[MultiAgentHarnessRequest]):
    def __init__(self, coordinator: "MultiAgentCoordinator") -> None:
        self.coordinator = coordinator
        super().__init__()

    def _build_initial_state(self, request: MultiAgentHarnessRequest) -> MultiAgentState:
        return {
            "query": request.query,
            "intent": request.intent,
            "slots": request.slots,
            "mode": request.mode,
            "session_id": request.session_id,
            "trace_id": request.trace_id,
            "task_config": request.task_config,
            "history": list(request.history),
            "prior_search_result": request.prior_search_result,
            "flow": list(request.flow),
            "artifacts": {},
        }

    def _extract_result(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return dict(state.get("artifacts") or {})

    def _build_graph(self) -> Any:
        if StateGraph is None:
            return None
        graph = StateGraph(MultiAgentState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("search", self._search_node)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("debate", self._debate_node)
        graph.add_node("write", self._write_node)
        graph.add_node("coder", self._coder_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "analyze")
        graph.add_edge("analyze", "debate")
        graph.add_edge("debate", "write")
        graph.add_edge("write", "coder")
        graph.add_edge("coder", END)
        return graph.compile()

    def _plan_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "plan" not in state.get("flow", []):
            return {}
        research_plan = self.coordinator.planner_agent.run(
            state["query"],
            state["intent"],
            state["slots"],
            state["trace_id"],
        )
        return self.merge_artifacts(state, research_plan=research_plan)

    def _search_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "search" not in state.get("flow", []):
            return {}
        search_result = self.coordinator.search_agent.run(
            state["query"],
            state["intent"],
            state["slots"],
            state.get("history", []),
            state["trace_id"],
            session_id=state.get("session_id", ""),
            prior_search_result=state.get("prior_search_result"),
        )
        return self.merge_artifacts(state, search_result=search_result)

    def _analyze_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "analyze" not in state.get("flow", []):
            return {}
        search_result = state.get("search_result")
        if search_result is None:
            return {}
        analysis_limit = self.coordinator._resolve_analysis_limit(
            flow=state["flow"],
            current_index=self.coordinator._flow_index(state["flow"], "analyze"),
            query=state["query"],
            task_config=state.get("task_config"),
        )
        analyses = self.coordinator.analyze_agent.run(
            search_result.papers,
            state["trace_id"],
            max_items=analysis_limit,
            slots=state.get("slots") or {},
            user_id=state.get("session_id", ""),
        )
        return self.merge_artifacts(state, analyses=analyses)

    def _debate_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "debate" not in state.get("flow", []):
            return {}
        debate = self.coordinator.debate_agent.run(
            state["query"],
            state.get("analyses") or [],
            state["trace_id"],
            task_config=state.get("task_config"),
        )
        return self.merge_artifacts(state, debate=debate)

    def _write_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "write" not in state.get("flow", []):
            return {}
        answer = self.coordinator.write_agent.run(
            state["intent"],
            state["query"],
            state.get("research_plan"),
            state.get("search_result"),
            state.get("analyses") or [],
            state.get("debate"),
            state["trace_id"],
        )
        return self.merge_artifacts(state, answer=answer)

    def _coder_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "coder" not in state.get("flow", []):
            return {}
        answer = self.coordinator.coder_agent.run(
            state["query"],
            state.get("analyses") or [],
            state["trace_id"],
        )
        return self.merge_artifacts(state, answer=answer)

    def _execute_sequential(self, state: MultiAgentState) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {}
        research_plan: ResearchPlan | None = None
        search_result: SearchResult | None = None
        analyses: List[PaperAnalysis] = []
        debate: DebateResult | None = None
        flow = state.get("flow") or []
        task_config = state.get("task_config")

        for index, step in enumerate(flow):
            if step == "plan":
                research_plan = self.coordinator.planner_agent.run(
                    state["query"],
                    state["intent"],
                    state["slots"],
                    state["trace_id"],
                )
                artifacts["research_plan"] = research_plan
            elif step == "search":
                search_result = self.coordinator.search_agent.run(
                    state["query"],
                    state["intent"],
                    state["slots"],
                    state.get("history", []),
                    state["trace_id"],
                    session_id=state.get("session_id", ""),
                    prior_search_result=state.get("prior_search_result"),
                )
                artifacts["search_result"] = search_result
            elif step == "analyze" and search_result is not None:
                analysis_limit = self.coordinator._resolve_analysis_limit(
                    flow=flow,
                    current_index=index,
                    query=state["query"],
                    task_config=task_config,
                )
                analyses = self.coordinator.analyze_agent.run(
                    search_result.papers,
                    state["trace_id"],
                    max_items=analysis_limit,
                    slots=state.get("slots") or {},
                    user_id=state.get("session_id", ""),
                )
                artifacts["analyses"] = analyses
            elif step == "debate":
                debate = self.coordinator.debate_agent.run(
                    state["query"],
                    analyses,
                    state["trace_id"],
                    task_config=task_config,
                )
                artifacts["debate"] = debate
            elif step == "write":
                artifacts["answer"] = self.coordinator.write_agent.run(
                    state["intent"],
                    state["query"],
                    research_plan,
                    search_result,
                    analyses,
                    debate,
                    state["trace_id"],
                )
            elif step == "coder":
                artifacts["answer"] = self.coordinator.coder_agent.run(
                    state["query"],
                    analyses,
                    state["trace_id"],
                )

        return artifacts
