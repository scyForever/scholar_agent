from __future__ import annotations

from dataclasses import asdict

from src.core.llm import LLMManager
from src.core.models import ReasoningResult
from src.whitebox.tracer import WhiteboxTracer


class ReasoningEngine:
    def __init__(self, llm: LLMManager | None = None, tracer: WhiteboxTracer | None = None) -> None:
        self.llm = llm or LLMManager()
        self.tracer = tracer or WhiteboxTracer()

    def reason(self, query: str, context: str, mode: str = "auto", trace_id: str | None = None) -> ReasoningResult:
        if mode == "auto":
            mode = self._auto_mode(query)

        methods = {
            "direct": self._direct_answer,
            "cot": self._chain_of_thought,
            "react": self._react_loop,
            "tot": self._tree_of_thought,
            "debate": self._debate_reasoning,
            "reflection": self._reflection_loop,
            "cove": self._chain_of_verification,
        }
        result = methods[mode](query, context)
        if trace_id:
            self.tracer.trace_step(trace_id, f"reasoning:{mode}", {"query": query}, asdict(result))
        return result

    def _auto_mode(self, query: str) -> str:
        if any(token in query for token in ("对比", "比较", "区别")):
            return "debate"
        if any(token in query for token in ("综述", "survey", "路线")):
            return "reflection"
        if len(query) > 50:
            return "cot"
        return "direct"

    def _direct_answer(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(f"问题：{query}\n上下文：{context}\n请直接给出结构化回答。")
        return ReasoningResult(mode="direct", answer=answer, steps=["整理上下文", "直接回答"], confidence=0.6)

    def _chain_of_thought(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(
            f"问题：{query}\n上下文：{context}\n请先给出简洁的推理步骤，再给出答案。",
        )
        return ReasoningResult(mode="cot", answer=answer, steps=["拆解问题", "逐步推理", "形成结论"], confidence=0.72)

    def _react_loop(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(
            f"任务：{query}\n可用上下文：{context}\n请按照 Observation -> Action -> Conclusion 给出回答。",
        )
        return ReasoningResult(mode="react", answer=answer, steps=["观察", "行动规划", "结论"], confidence=0.68)

    def _tree_of_thought(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(
            f"问题：{query}\n上下文：{context}\n请给出3条思路分支，评估后选择最优方案。",
        )
        return ReasoningResult(mode="tot", answer=answer, steps=["生成多分支", "比较分支", "选择最优"], confidence=0.74)

    def _debate_reasoning(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(
            f"问题：{query}\n材料：{context}\n请分别从支持和反对两个角度分析，再给出综合判断。",
        )
        return ReasoningResult(mode="debate", answer=answer, steps=["正方观点", "反方观点", "综合结论"], confidence=0.78)

    def _reflection_loop(self, query: str, context: str) -> ReasoningResult:
        draft = self.llm.call(f"请先生成问题“{query}”的初稿，材料：{context}")
        final = self.llm.call(f"请审阅并优化以下初稿，使其更严谨、更完整：\n{draft}")
        return ReasoningResult(mode="reflection", answer=final, steps=["初稿生成", "反思审阅", "优化输出"], confidence=0.8)

    def _chain_of_verification(self, query: str, context: str) -> ReasoningResult:
        draft = self.llm.call(f"问题：{query}\n材料：{context}\n先给出一个答案。")
        verification = self.llm.call(f"请针对以下答案提出3个验证问题并回答：\n{draft}")
        final = self.llm.call(f"请结合原答案与验证结果给出修正版。\n原答案：{draft}\n验证：{verification}")
        return ReasoningResult(
            mode="cove",
            answer=final,
            steps=["生成答案", "提出验证问题", "修正结论"],
            confidence=0.82,
        )
