from __future__ import annotations

from typing import Any, Dict, List

from src.agents.research_agents import ResearchMemoryAgent, ResearchReadingAgent
from src.core.llm import LLMManager
from src.core.models import Paper, PaperAnalysis
from src.prompt_templates.manager import PromptTemplateManager
from src.whitebox.tracer import WhiteboxTracer


class AnalyzeAgent:
    def __init__(
        self,
        llm: LLMManager,
        templates: PromptTemplateManager,
        tracer: WhiteboxTracer,
        reading_agent: ResearchReadingAgent | None = None,
        memory_agent: ResearchMemoryAgent | None = None,
    ) -> None:
        self.llm = llm
        self.templates = templates
        self.tracer = tracer
        self.reading_agent = reading_agent
        self.memory_agent = memory_agent

    def run(
        self,
        papers: List[Paper],
        trace_id: str,
        max_items: int | None = None,
        slots: Dict[str, Any] | None = None,
        user_id: str = "",
    ) -> List[PaperAnalysis]:
        analyses: List[PaperAnalysis] = []
        capped = 5 if max_items is None else max(0, min(max_items, 5))
        selected_papers = papers[:capped]
        resolved_slots = dict(slots or {})
        pdf_path = str(resolved_slots.get("pdf_path") or "")
        focus = str(resolved_slots.get("focus") or "")
        stage_token = self.llm.bind_stage("analyze")
        try:
            for paper in selected_papers:
                deep_context = ""
                if pdf_path and self.reading_agent is not None:
                    reading_payload = self.reading_agent.build_analysis_context(
                        paper,
                        pdf_path=pdf_path,
                        focus=focus,
                        trace_id=trace_id,
                    )
                    sections = reading_payload.get("sections") or []
                    chunks = reading_payload.get("chunks") or []
                    deep_context = "\n".join(
                        [
                            "深度阅读材料：",
                            *[
                                f"- {section.get('heading', 'section')}: {str(section.get('text') or '')[:800]}"
                                for section in sections[:3]
                            ],
                            *[f"- {str(chunk)[:800]}" for chunk in chunks[:4]],
                        ]
                    ).strip()
                prompt = self.templates.render(
                    "paper_analysis",
                    title=paper.title,
                    abstract=paper.abstract,
                    context=(
                        f"来源：{paper.source}, 年份：{paper.year}, 引用数：{paper.citations}"
                        + (f"\n{deep_context}" if deep_context else "")
                    ),
                )
                raw = self.llm.call(prompt, purpose="论文分析", budgeted=True)
                analysis = PaperAnalysis(
                    paper=paper,
                    summary=raw[:500],
                    contributions=self._extract_lines(raw, "贡献"),
                    methods=self._extract_lines(raw, "方法"),
                    findings=self._extract_lines(raw, "发现"),
                    limitations=self._extract_lines(raw, "局限"),
                    raw_analysis=raw,
                )
                analyses.append(analysis)
                if user_id and self.memory_agent is not None:
                    highlights = [*analysis.contributions[:2], *analysis.findings[:2]]
                    self.memory_agent.remember_paper(
                        user_id,
                        paper,
                        analysis.summary,
                        highlights=highlights,
                        trace_id=trace_id,
                    )
        finally:
            self.llm.reset_stage(stage_token)
        self.tracer.trace_step(
            trace_id,
            "analyze",
            {"papers": [paper.title for paper in selected_papers]},
            {
                "count": len(analyses),
                "analysis_limit": capped,
                "deep_read_used": bool(pdf_path),
                "focus": focus,
            },
        )
        return analyses

    def _extract_lines(self, text: str, keyword: str) -> List[str]:
        lines = []
        for line in text.splitlines():
            clean = line.strip("-* 0123456789.")
            if keyword in clean:
                lines.append(clean)
        return lines[:5]
