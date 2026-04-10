from __future__ import annotations

from typing import Any, Dict

from src.core.models import ExecutionMode
from src.harness.base import BaseHarness
from src.harness.contracts import RuntimeHarnessRequest
from src.pipeline.state import RuntimeState

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover
    END = None
    START = None
    StateGraph = None


class RuntimeHarness(BaseHarness[RuntimeHarnessRequest]):
    def __init__(
        self,
        *,
        multi_agent: Any,
        reasoning: Any,
        quality: Any,
        tracer: Any,
    ) -> None:
        self.multi_agent = multi_agent
        self.reasoning = reasoning
        self.quality = quality
        self.tracer = tracer
        super().__init__()

    def _build_initial_state(self, request: RuntimeHarnessRequest) -> RuntimeState:
        return {
            "query": request.query,
            "intent": request.intent,
            "slots": request.slots,
            "session_id": request.session_id,
            "trace_id": request.trace_id,
            "task_config": request.task_config,
            "history": list(request.history),
            "memory_context": request.memory_context,
            "prior_search_result": request.prior_search_result,
            "execution_mode": request.execution_mode,
            "enable_quality_enhance": request.enable_quality_enhance,
            "artifacts": {},
        }

    def _extract_result(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "answer": str(state.get("answer") or ""),
            "artifacts": dict(state.get("artifacts") or {}),
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
            session_id=state.get("session_id", ""),
            trace_id=state["trace_id"],
            task_config=state.get("task_config"),
            history=state.get("history") or [],
            prior_search_result=state.get("prior_search_result"),
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
            session_id=state.get("session_id", ""),
            trace_id=state["trace_id"],
            task_config=state.get("task_config"),
            history=state.get("history") or [],
            prior_search_result=state.get("prior_search_result"),
        )
        answer = str(artifacts.get("answer") or "")

        if not answer:
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

        return {
            "answer": answer,
            "artifacts": artifacts,
        }
