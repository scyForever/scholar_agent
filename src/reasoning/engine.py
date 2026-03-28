from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List

from src.core.llm import LLMManager
from src.core.models import ReasoningResult
from src.whitebox.tracer import WhiteboxTracer


class ReasoningEngine:
    def __init__(self, llm: LLMManager | None = None, tracer: WhiteboxTracer | None = None) -> None:
        self.llm = llm or LLMManager()
        self.tracer = tracer or WhiteboxTracer()

    def reason(
        self,
        query: str,
        context: str,
        mode: str = "auto",
        trace_id: str | None = None,
        preferred_modes: Iterable[str] | None = None,
    ) -> ReasoningResult:
        methods = {
            "direct": self._direct_answer,
            "cot": self._chain_of_thought,
            "react": self._react_loop,
            "tot": self._tree_of_thought,
            "debate": self._debate_reasoning,
            "reflection": self._reflection_loop,
            "cove": self._chain_of_verification,
        }
        resolved_mode = self._resolve_mode(query, mode, preferred_modes, methods.keys())
        result = methods[resolved_mode](query, context)
        if trace_id:
            self.tracer.trace_step(
                trace_id,
                f"reasoning:{resolved_mode}",
                {
                    "query": query,
                    "requested_mode": mode,
                    "preferred_modes": list(self._normalize_modes(preferred_modes, methods.keys())),
                },
                asdict(result),
            )
        return result

    def estimate_llm_calls(
        self,
        query: str,
        mode: str = "auto",
        preferred_modes: Iterable[str] | None = None,
    ) -> int:
        methods = {"direct", "cot", "react", "tot", "debate", "reflection", "cove"}
        resolved_mode = self._resolve_mode(query, mode, preferred_modes, methods)
        if resolved_mode == "reflection":
            return 2
        if resolved_mode == "cove":
            return 3
        return 1

    def _resolve_mode(
        self,
        query: str,
        mode: str,
        preferred_modes: Iterable[str] | None,
        supported_modes: Iterable[str],
    ) -> str:
        normalized_supported = set(supported_modes)
        normalized_preferred = self._normalize_modes(preferred_modes, normalized_supported)
        if mode != "auto":
            if mode not in normalized_supported:
                raise ValueError(f"Unsupported reasoning mode: {mode}")
            if normalized_preferred and mode not in normalized_preferred:
                return normalized_preferred[0]
            return mode
        return self._auto_mode(query, normalized_preferred)

    def _normalize_modes(
        self,
        preferred_modes: Iterable[str] | None,
        supported_modes: Iterable[str],
    ) -> List[str]:
        supported = set(supported_modes)
        normalized: List[str] = []
        for mode in preferred_modes or []:
            if mode in supported and mode not in normalized:
                normalized.append(mode)
        return normalized

    def _auto_mode(self, query: str, preferred_modes: Iterable[str] | None = None) -> str:
        allowed = list(preferred_modes or ["direct", "cot", "react", "tot", "debate", "reflection", "cove"])
        lowered = query.lower()
        candidates: List[str] = []

        if any(token in query for token in ("对比", "比较", "区别")):
            candidates.extend(["debate", "cot", "direct"])
        if any(token in query for token in ("综述", "survey", "路线")):
            candidates.extend(["reflection", "cot", "direct"])
        if any(token in query for token in ("实现", "代码", "步骤", "怎么做", "流程", "方案")):
            candidates.extend(["react", "cot", "direct"])
        if any(token in query for token in ("验证", "核实", "检查", "自洽")):
            candidates.extend(["cove", "cot", "direct"])
        if any(token in lowered for token in ("trade-off", "branch", "branches", "path")):
            candidates.extend(["tot", "cot", "direct"])
        if len(query) > 50:
            candidates.extend(["cot", "direct"])
        else:
            candidates.extend(["direct", "cot"])

        for candidate in candidates:
            if candidate in allowed:
                return candidate
        return allowed[0] if allowed else "direct"

    def _direct_answer(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(
            f"问题：{query}\n上下文：{context}\n请直接给出结构化回答。",
            purpose="直接回答",
            budgeted=True,
        )
        return ReasoningResult(mode="direct", answer=answer, steps=["整理上下文", "直接回答"], confidence=0.6)

    def _chain_of_thought(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(
            f"问题：{query}\n上下文：{context}\n请先给出简洁的推理步骤，再给出答案。",
            purpose="链式推理",
            budgeted=True,
        )
        return ReasoningResult(mode="cot", answer=answer, steps=["拆解问题", "逐步推理", "形成结论"], confidence=0.72)

    def _react_loop(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(
            f"任务：{query}\n可用上下文：{context}\n请按照 Observation -> Action -> Conclusion 给出回答。",
            purpose="ReAct推理",
            budgeted=True,
        )
        return ReasoningResult(mode="react", answer=answer, steps=["观察", "行动规划", "结论"], confidence=0.68)

    def _tree_of_thought(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(
            f"问题：{query}\n上下文：{context}\n请给出3条思路分支，评估后选择最优方案。",
            purpose="树状推理",
            budgeted=True,
        )
        return ReasoningResult(mode="tot", answer=answer, steps=["生成多分支", "比较分支", "选择最优"], confidence=0.74)

    def _debate_reasoning(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(
            f"问题：{query}\n材料：{context}\n请分别从支持和反对两个角度分析，再给出综合判断。",
            purpose="辩论推理",
            budgeted=True,
        )
        return ReasoningResult(mode="debate", answer=answer, steps=["正方观点", "反方观点", "综合结论"], confidence=0.78)

    def _reflection_loop(self, query: str, context: str) -> ReasoningResult:
        draft = self.llm.call(
            f"请先生成问题“{query}”的初稿，材料：{context}",
            purpose="反思推理-初稿",
            budgeted=True,
        )
        final = self.llm.call(
            f"请审阅并优化以下初稿，使其更严谨、更完整：\n{draft}",
            purpose="反思推理-修订",
            budgeted=True,
        )
        return ReasoningResult(mode="reflection", answer=final, steps=["初稿生成", "反思审阅", "优化输出"], confidence=0.8)

    def _chain_of_verification(self, query: str, context: str) -> ReasoningResult:
        draft = self.llm.call(
            f"问题：{query}\n材料：{context}\n先给出一个答案。",
            purpose="验证链-初稿",
            budgeted=True,
        )
        verification = self.llm.call(
            f"请针对以下答案提出3个验证问题并回答：\n{draft}",
            purpose="验证链-验证",
            budgeted=True,
        )
        final = self.llm.call(
            f"请结合原答案与验证结果给出修正版。\n原答案：{draft}\n验证：{verification}",
            purpose="验证链-修订",
            budgeted=True,
        )
        return ReasoningResult(
            mode="cove",
            answer=final,
            steps=["生成答案", "提出验证问题", "修正结论"],
            confidence=0.82,
        )
