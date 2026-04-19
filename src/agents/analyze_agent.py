from __future__ import annotations

from typing import Any, Dict, List

from src.agents.research_agents import ResearchMemoryAgent, ResearchReadingAgent
from src.core.llm import LLMManager
from src.core.models import Paper, PaperAnalysis
from src.prompt_templates.manager import PromptTemplateManager
from src.tools.research_search_tool import FUSION_SOURCE_COUNT_KEY
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
        prioritized_papers = self._prioritize_papers_for_analysis(papers, resolved_slots)
        selected_papers = [item["paper"] for item in prioritized_papers[:capped]]
        pdf_path = str(resolved_slots.get("pdf_path") or "")
        focus = str(resolved_slots.get("focus") or "")
        stage_token = self.llm.bind_stage("analyze")
        try:
            for item in prioritized_papers[:capped]:
                paper = item["paper"]
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
                        "证据优先级："
                        + self._priority_summary(item)
                        + "\n"
                        + f"来源：{paper.source}, 年份：{paper.year}, 引用数：{paper.citations}"
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
                "evidence_priority": [
                    {
                        "title": item["paper"].title,
                        "score": round(float(item["score"]), 3),
                        "reasons": list(item["reasons"]),
                    }
                    for item in prioritized_papers[:capped]
                ],
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

    def _prioritize_papers_for_analysis(
        self,
        papers: List[Paper],
        slots: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        prioritized = []
        for paper in papers:
            score, reasons = self._paper_evidence_priority(paper, slots)
            prioritized.append(
                {
                    "paper": paper,
                    "score": score,
                    "reasons": reasons,
                }
            )
        prioritized.sort(
            key=lambda item: (
                float(item["score"]),
                float(item["paper"].score or 0.0),
                int(item["paper"].citations or 0),
                int(item["paper"].year or 0),
            ),
            reverse=True,
        )
        return prioritized

    def _paper_evidence_priority(
        self,
        paper: Paper,
        slots: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        score = float(paper.score or 0.0) * 3.0
        reasons: List[str] = []
        metadata = dict(paper.metadata or {})
        abstract_text = str(paper.abstract or "").strip()
        text_blob = f"{paper.title} {paper.abstract}".lower()
        full_text_ready = bool((paper.pdf_url or paper.full_text_url or paper.html_url or "").strip())
        source_count = max(int(metadata.get(FUSION_SOURCE_COUNT_KEY, 1) or 1), 1)
        outline_depth = str(slots.get("outline_depth") or "").strip()
        organization_style = str(slots.get("organization_style") or "").strip()
        required_sections = [
            str(item).strip()
            for item in (slots.get("required_sections") or [])
            if str(item).strip()
        ]

        if abstract_text:
            score += 1.0
            reasons.append("摘要完整")
        if len(abstract_text) >= 180:
            score += 0.4
        if full_text_ready:
            score += 1.5
            reasons.append("可直达全文或 PDF")
        if paper.open_access:
            score += 0.8
            reasons.append("开放获取")
        if source_count > 1:
            score += min(source_count, 3) * 0.6
            reasons.append("跨源合并后元数据更完整")
        if paper.year is not None:
            score += 0.3
        score += min(int(paper.citations or 0), 200) / 80.0

        if outline_depth == "deep":
            if full_text_ready:
                score += 1.2
            score += 0.6 if len(abstract_text) >= 240 else 0.0
            reasons.append("详细写作优先深证据论文")

        if required_sections:
            if full_text_ready:
                score += 1.0
            if self._contains_any(text_blob, ["survey", "review", "overview", "taxonomy", "benchmark", "挑战", "综述"]):
                score += 0.8
                reasons.append("更适合支撑综述章节组织")

        if organization_style == "timeline" and paper.year is not None:
            score += 0.9
            reasons.append("时间线组织需要年份锚点")
        elif organization_style == "method" and self._contains_any(text_blob, ["method", "framework", "architecture", "approach", "model", "algorithm"]):
            score += 0.8
            reasons.append("方法线组织优先方法型论文")
        elif organization_style == "application" and self._contains_any(text_blob, ["application", "case study", "benchmark", "dataset", "deployment"]):
            score += 0.8
            reasons.append("应用线组织优先场景型论文")
        elif organization_style == "topic" and self._contains_any(text_blob, ["survey", "review", "overview", "taxonomy"]):
            score += 0.5
            reasons.append("主题线组织优先综述性论文")

        return score, list(dict.fromkeys(reasons))

    def _priority_summary(self, item: Dict[str, Any]) -> str:
        reasons = [str(reason).strip() for reason in item.get("reasons") or [] if str(reason).strip()]
        if not reasons:
            return f"综合优先级 {float(item.get('score') or 0.0):.2f}"
        return f"综合优先级 {float(item.get('score') or 0.0):.2f}；" + "；".join(reasons[:4])

    def _contains_any(self, text: str, keywords: List[str]) -> bool:
        lowered = str(text or "").lower()
        return any(str(keyword).lower() in lowered for keyword in keywords if str(keyword).strip())
