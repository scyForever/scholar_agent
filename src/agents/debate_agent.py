from __future__ import annotations

from dataclasses import asdict
from typing import List

from src.core.models import DebateResult, PaperAnalysis
from src.planning.task_hierarchy import TaskConfig
from src.reasoning.engine import ReasoningEngine
from src.whitebox.tracer import WhiteboxTracer


class DebateAgent:
    def __init__(self, reasoning: ReasoningEngine, tracer: WhiteboxTracer) -> None:
        self.reasoning = reasoning
        self.tracer = tracer

    def run(
        self,
        query: str,
        analyses: List[PaperAnalysis],
        trace_id: str,
        task_config: TaskConfig | None = None,
    ) -> DebateResult:
        materials = "\n\n".join(
            f"论文：{item.paper.title}\n摘要：{item.summary}\n贡献：{'；'.join(item.contributions)}"
            for item in analyses
        )
        base_modes = task_config.reasoning_modes if task_config is not None else []
        preferred_modes = list(dict.fromkeys(["debate", *base_modes]))
        result = self.reasoning.reason(
            query,
            materials,
            mode="debate",
            trace_id=trace_id,
            preferred_modes=preferred_modes,
            stage="debate",
        )
        debate = DebateResult(
            question=query,
            thesis="围绕研究问题的多视角综合判断",
            supporting_points=[item.paper.title for item in analyses[:3]],
            counter_points=[item.paper.title for item in analyses[3:5]],
            synthesis=result.answer,
        )
        self.tracer.trace_step(trace_id, "debate", {"query": query}, asdict(debate))
        return debate
