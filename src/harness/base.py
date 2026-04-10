from __future__ import annotations

from typing import Any, Dict, Generic, TypeVar

RequestT = TypeVar("RequestT")


class BaseHarness(Generic[RequestT]):
    def __init__(self) -> None:
        self.graph = self._build_graph()

    def uses_langgraph(self) -> bool:
        return self.graph is not None

    def execute(self, request: RequestT) -> Dict[str, Any]:
        initial_state = self._build_initial_state(request)
        if self.graph is None:
            return self._execute_sequential(initial_state)
        result = self.graph.invoke(initial_state)
        return self._extract_result(result)

    def _build_graph(self) -> Any:
        return None

    def _build_initial_state(self, request: RequestT) -> Dict[str, Any]:
        raise NotImplementedError

    def _execute_sequential(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def _extract_result(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def merge_artifacts(self, state: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
        artifacts = dict(state.get("artifacts") or {})
        artifacts.update(updates)
        if updates.get("answer"):
            artifacts["answer"] = updates["answer"]
        return {**updates, "artifacts": artifacts}
