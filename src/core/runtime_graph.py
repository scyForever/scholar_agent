from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from src.agents.multi_agent import MultiAgentCoordinator
from src.core.models import ExecutionMode
from src.planning.task_hierarchy import TaskConfig
from src.quality.enhancer import QualityEnhancer
from src.reasoning.engine import ReasoningEngine
from src.whitebox.tracer import WhiteboxTracer

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover
    END = None
    START = None
    StateGraph = None


class RuntimeState(TypedDict, total=False):
    query: str
    intent: str
    slots: Dict[str, Any]
    trace_id: str
    task_config: TaskConfig | None
    history: List[Dict[str, str]]
    memory_context: str
    execution_mode: ExecutionMode
    enable_quality_enhance: bool
    answer: str
    artifacts: Dict[str, Any]


class AgentRuntimeGraph:
    def __init__(
        self,
        multi_agent: MultiAgentCoordinator,
        reasoning: ReasoningEngine,
        quality: QualityEnhancer,
        tracer: WhiteboxTracer,
    ) -> None:
        self.multi_agent = multi_agent
        self.reasoning = reasoning
        self.quality = quality
        self.tracer = tracer
        self.graph = self._build_graph()

    def uses_langgraph(self) -> bool:
        return self.graph is not None

    def execute(
        self,
        *,
        query: str,
        intent: str,
        slots: Dict[str, Any],
        trace_id: str,
        task_config: TaskConfig,
        history: List[Dict[str, str]],
        memory_context: str,
        execution_mode: ExecutionMode,
        enable_quality_enhance: bool,
    ) -> Dict[str, Any]:
        initial_state: RuntimeState = {
            "query": query,
            "intent": intent,
            "slots": slots,
            "trace_id": trace_id,
            "task_config": task_config,
            "history": history,
            "memory_context": memory_context,
            "execution_mode": execution_mode,
            "enable_quality_enhance": enable_quality_enhance,
            "artifacts": {},
        }
        if self.graph is None:
            return self._execute_sequential(initial_state)
        result = self.graph.invoke(initial_state)
        return {
            "answer": str(result.get("answer") or ""),
            "artifacts": dict(result.get("artifacts") or {}),
        }

    def _build_graph(self) -> Any:
        if StateGraph is None:
            return None
        graph = StateGraph(RuntimeState)
        graph.add_node("multi_agent", self._multi_agent_node)
        graph.add_node("reasoning", self._reasoning_node)
        graph.add_node("quality", self._quality_node)
        graph.add_edge(START, "multi_agent")
        graph.add_conditional_edges(
            "multi_agent",
            self._route_after_generation,
            {
                "reasoning": "reasoning",
                "quality": "quality",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "reasoning",
            self._route_after_generation,
            {
                "quality": "quality",
                "end": END,
            },
        )
        graph.add_edge("quality", END)
        return graph.compile()

    def _multi_agent_node(self, state: RuntimeState) -> Dict[str, Any]:
        artifacts = self.multi_agent.execute(
            query=state["query"],
            intent=state["intent"],
            slots=state["slots"],
            mode=state["execution_mode"],
            trace_id=state["trace_id"],
            task_config=state.get("task_config"),
            history=state.get("history") or [],
        )
        answer = str(artifacts.get("answer") or "")
        return {
            "answer": answer,
            "artifacts": dict(artifacts),
        }

    def _reasoning_node(self, state: RuntimeState) -> Dict[str, Any]:
        task_config = state.get("task_config")
        preferred_modes = task_config.reasoning_modes if task_config is not None else None
        reasoning = self.reasoning.reason(
            state["query"],
            state.get("memory_context") or "",
            mode="auto",
            trace_id=state["trace_id"],
            preferred_modes=preferred_modes,
            stage="reasoning",
        )
        artifacts = dict(state.get("artifacts") or {})
        artifacts["reasoning"] = reasoning
        return {
            "answer": reasoning.answer,
            "artifacts": artifacts,
        }

    def _quality_node(self, state: RuntimeState) -> Dict[str, Any]:
        base_answer = str(state.get("answer") or "")
        artifacts = dict(state.get("artifacts") or {})
        try:
            moa_result = self.quality.self_moa(state["query"], base_answer)
            verification = self.quality.mpsc_verify(state["query"], moa_result.answer)
            answer = moa_result.answer
            artifacts["moa"] = moa_result
            artifacts["verification"] = verification
            self.tracer.trace_step(
                state["trace_id"],
                "quality",
                {"query": state["query"]},
                {
                    "verification": verification.verdict,
                    "moa_errors": moa_result.errors,
                    "verification_errors": verification.errors,
                },
            )
            return {
                "answer": answer,
                "artifacts": artifacts,
            }
        except Exception as exc:
            artifacts["quality_error"] = f"{type(exc).__name__}: {exc}"
            self.tracer.trace_step(
                state["trace_id"],
                "quality",
                {"query": state["query"]},
                {
                    "verification": "failed_but_preserved_answer",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return {
                "answer": base_answer,
                "artifacts": artifacts,
            }

    def _route_after_generation(self, state: RuntimeState) -> str:
        if not state.get("answer"):
            return "reasoning"
        if state.get("enable_quality_enhance") and state.get("execution_mode") == ExecutionMode.FULL:
            return "quality"
        return "end"

    def _execute_sequential(self, state: RuntimeState) -> Dict[str, Any]:
        artifacts = self.multi_agent.execute(
            query=state["query"],
            intent=state["intent"],
            slots=state["slots"],
            mode=state["execution_mode"],
            trace_id=state["trace_id"],
            task_config=state.get("task_config"),
            history=state.get("history") or [],
        )
        answer = str(artifacts.get("answer") or "")

        if not answer:
            preferred_modes = None
            task_config = state.get("task_config")
            if task_config is not None:
                preferred_modes = task_config.reasoning_modes
            reasoning = self.reasoning.reason(
                state["query"],
                state.get("memory_context") or "",
                mode="auto",
                trace_id=state["trace_id"],
                preferred_modes=preferred_modes,
                stage="reasoning",
            )
            answer = reasoning.answer
            artifacts["reasoning"] = reasoning

        if state.get("enable_quality_enhance") and state.get("execution_mode") == ExecutionMode.FULL:
            base_answer = answer
            try:
                moa_result = self.quality.self_moa(state["query"], answer)
                verification = self.quality.mpsc_verify(state["query"], moa_result.answer)
                answer = moa_result.answer
                artifacts["moa"] = moa_result
                artifacts["verification"] = verification
                self.tracer.trace_step(
                    state["trace_id"],
                    "quality",
                    {"query": state["query"]},
                    {
                        "verification": verification.verdict,
                        "moa_errors": moa_result.errors,
                        "verification_errors": verification.errors,
                    },
                )
            except Exception as exc:
                answer = base_answer
                artifacts["quality_error"] = f"{type(exc).__name__}: {exc}"
                self.tracer.trace_step(
                    state["trace_id"],
                    "quality",
                    {"query": state["query"]},
                    {
                        "verification": "failed_but_preserved_answer",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )

        artifacts["answer"] = answer
        return {"answer": answer, "artifacts": artifacts}
