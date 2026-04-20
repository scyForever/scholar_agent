from __future__ import annotations

from typing import List

from src.core.llm import LLMManager
from src.core.models import PaperAnalysis
from src.prompt_templates.manager import PromptTemplateManager
from src.whitebox.tracer import WhiteboxTracer


class CoderAgent:
    def __init__(
        self, llm: LLMManager, templates: PromptTemplateManager, tracer: WhiteboxTracer
    ) -> None:
        self.llm = llm
        self.templates = templates
        self.tracer = tracer

    def run(self, query: str, analyses: List[PaperAnalysis], trace_id: str) -> str:
        materials = "\n".join(
            f"- {analysis.paper.title}: {analysis.summary}" for analysis in analyses
        )
        prompt = self.templates.render(
            "code_generation", topic=query, materials=materials
        )
        answer = self.llm.call(prompt, purpose="代码生成", budgeted=True)
        self.tracer.trace_step(
            trace_id, "coder", {"query": query}, {"answer_preview": answer[:500]}
        )
        return answer
