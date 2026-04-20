from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from src.core.models import DebateResult, ExecutionMode, PaperAnalysis, ResearchPlan, SearchResult
from src.pipeline.state import MultiAgentState

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover
    END = None
    START = None
    StateGraph = None

if TYPE_CHECKING:
    from src.agents.multi_agent import MultiAgentCoordinator
    from src.planning.task_hierarchy import TaskConfig


class MultiAgentPipeline:
    def __init__(self, coordinator: "MultiAgentCoordinator") -> None:
        self.coordinator = coordinator
        self.graph = self._build_graph()

    def uses_langgraph(self) -> bool:
        return self.graph is not None

    def execute(
        self,
        *,
        query: str,
        intent: str,
        slots: Dict[str, Any],
        mode: ExecutionMode,
        trace_id: str,
        flow: List[str],
        task_config: "TaskConfig | None" = None,
        history: List[Dict[str, str]] | None = None,
        session_id: str = "",
        prior_search_result: SearchResult | None = None,
    ) -> Dict[str, Any]:
        initial_state: MultiAgentState = {
            "query": query,
            "intent": intent,
            "slots": slots,
            "mode": mode,
            "session_id": session_id,
            "trace_id": trace_id,
            "task_config": task_config,
            "history": history or [],
            "prior_search_result": prior_search_result,
            "flow": flow,
            "artifacts": {},
        }
        if self.graph is None:
            return self._execute_sequential(initial_state)
        result = self.graph.invoke(initial_state)
        return dict(result.get("artifacts") or {})

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
        return self._state_update(state, research_plan=research_plan)

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
        return self._state_update(state, search_result=search_result)

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
        return self._state_update(state, analyses=analyses)

    def _debate_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "debate" not in state.get("flow", []):
            return {}
        debate = self.coordinator.debate_agent.run(
            state["query"],
            state.get("analyses") or [],
            state["trace_id"],
            task_config=state.get("task_config"),
        )
        return self._state_update(state, debate=debate)

    def _write_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "write" not in state.get("flow", []):
            return {}
        answer = self.coordinator.write_agent.run(
            state["intent"],
            state["query"],
            state.get("slots") or {},
            state.get("research_plan"),
            state.get("search_result"),
            state.get("analyses") or [],
            state.get("debate"),
            state["trace_id"],
        )
        return self._state_update(state, answer=answer)

    def _coder_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "coder" not in state.get("flow", []):
            return {}
        answer = self.coordinator.coder_agent.run(
            state["query"],
            state.get("analyses") or [],
            state["trace_id"],
        )
        return self._state_update(state, answer=answer)

    def _state_update(self, state: MultiAgentState, **updates: Any) -> Dict[str, Any]:
        artifacts = dict(state.get("artifacts") or {})
        artifacts.update(updates)
        if "answer" in updates and updates["answer"]:
            artifacts["answer"] = updates["answer"]
        return {**updates, "artifacts": artifacts}

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
                    state.get("slots") or {},
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
