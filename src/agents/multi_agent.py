from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List

from config.settings import settings
from src.core.llm import LLMManager
from src.core.models import DebateResult, ExecutionMode, Paper, PaperAnalysis, SearchResult
from src.preprocessing.query_rewriter import QueryRewriter
from src.prompt_templates.manager import PromptTemplateManager
from src.rag.retriever import HybridRetriever
from src.reasoning.engine import ReasoningEngine
from src.tools import TOOL_REGISTRY
from src.whitelist.manager import WhitelistManager
from src.whitebox.tracer import WhiteboxTracer


class SearchAgent:
    def __init__(self, retriever: HybridRetriever, whitelist: WhitelistManager, tracer: WhiteboxTracer) -> None:
        self.retriever = retriever
        self.whitelist = whitelist
        self.tracer = tracer
        self.rewriter = QueryRewriter(retriever.llm)

    def run(
        self,
        query: str,
        intent: str,
        slots: Dict[str, Any],
        history: List[Dict[str, str]],
        trace_id: str,
    ) -> SearchResult:
        topic = str(slots.get("topic") or slots.get("paper_title") or query)
        time_range = str(slots.get("time_range") or "")
        max_results = int(slots.get("max_papers") or 12)
        rewrite_plan = self.rewriter.plan(topic, intent=intent)
        rewritten_queries = rewrite_plan.external_queries
        allowed_tools = [
            tool for tool in self.whitelist.allowed_tools("search_agent") if tool != "search_web"
        ]

        aggregated: Dict[str, Paper] = {}
        for tool_name in allowed_tools:
            for rewritten in rewritten_queries[:3]:
                try:
                    papers = TOOL_REGISTRY.call(
                        tool_name,
                        query=rewritten,
                        max_results=max_results // max(len(allowed_tools), 1) + 2,
                        time_range=time_range,
                    )
                except TypeError:
                    papers = TOOL_REGISTRY.call(tool_name, query=rewritten, max_results=max_results)
                for paper in papers:
                    key = (paper.title or "").strip().lower()
                    if not key:
                        continue
                    paper.score = self._paper_score(paper)
                    if key not in aggregated or paper.score > aggregated[key].score:
                        aggregated[key] = paper

        local_context = self.retriever.retrieve(
            topic,
            history,
            top_k=5,
            rewritten_queries=rewrite_plan.local_queries,
            rewrite_plan=rewrite_plan,
        )
        papers = sorted(
            aggregated.values(),
            key=lambda item: (item.score, item.citations, item.year or 0),
            reverse=True,
        )[:max_results]
        source_breakdown: Dict[str, int] = defaultdict(int)
        for paper in papers:
            source_breakdown[paper.source] += 1
        result = SearchResult(
            query=topic,
            papers=papers,
            total_found=len(aggregated),
            source_breakdown=dict(source_breakdown),
            rewritten_queries=rewritten_queries,
            trace={"local_rag": local_context},
        )
        self.tracer.trace_step(trace_id, "search", {"query": topic}, asdict(result))
        return result

    def _paper_score(self, paper: Paper) -> float:
        recency = float((paper.year or 0) / 3000)
        citations = min(paper.citations / 1000.0, 1.0)
        return 0.55 * citations + 0.45 * recency


class AnalyzeAgent:
    def __init__(self, llm: LLMManager, templates: PromptTemplateManager, tracer: WhiteboxTracer) -> None:
        self.llm = llm
        self.templates = templates
        self.tracer = tracer

    def run(self, papers: List[Paper], trace_id: str) -> List[PaperAnalysis]:
        analyses: List[PaperAnalysis] = []
        for paper in papers[:5]:
            prompt = self.templates.render(
                "paper_analysis",
                title=paper.title,
                abstract=paper.abstract,
                context=f"来源：{paper.source}, 年份：{paper.year}, 引用数：{paper.citations}",
            )
            raw = self.llm.call(prompt, purpose="论文分析")
            analyses.append(
                PaperAnalysis(
                    paper=paper,
                    summary=raw[:500],
                    contributions=self._extract_lines(raw, "贡献"),
                    methods=self._extract_lines(raw, "方法"),
                    findings=self._extract_lines(raw, "发现"),
                    limitations=self._extract_lines(raw, "局限"),
                    raw_analysis=raw,
                )
            )
        self.tracer.trace_step(trace_id, "analyze", {"papers": [paper.title for paper in papers[:5]]}, {"count": len(analyses)})
        return analyses

    def _extract_lines(self, text: str, keyword: str) -> List[str]:
        lines = []
        for line in text.splitlines():
            clean = line.strip("-* 0123456789.")
            if keyword in clean:
                lines.append(clean)
        return lines[:5]


class DebateAgent:
    def __init__(self, reasoning: ReasoningEngine, tracer: WhiteboxTracer) -> None:
        self.reasoning = reasoning
        self.tracer = tracer

    def run(self, query: str, analyses: List[PaperAnalysis], trace_id: str) -> DebateResult:
        materials = "\n\n".join(
            f"论文：{item.paper.title}\n摘要：{item.summary}\n贡献：{'；'.join(item.contributions)}"
            for item in analyses
        )
        result = self.reasoning.reason(query, materials, mode="debate", trace_id=trace_id)
        debate = DebateResult(
            question=query,
            thesis="围绕研究问题的多视角综合判断",
            supporting_points=[item.paper.title for item in analyses[:3]],
            counter_points=[item.paper.title for item in analyses[3:5]],
            synthesis=result.answer,
        )
        self.tracer.trace_step(trace_id, "debate", {"query": query}, asdict(debate))
        return debate


class WriteAgent:
    def __init__(self, llm: LLMManager, templates: PromptTemplateManager, tracer: WhiteboxTracer) -> None:
        self.llm = llm
        self.templates = templates
        self.tracer = tracer

    def run(
        self,
        intent: str,
        query: str,
        search_result: SearchResult | None,
        analyses: List[PaperAnalysis],
        debate: DebateResult | None,
        trace_id: str,
    ) -> str:
        if intent == "search_papers" and search_result is not None:
            lines = [f"共找到 {len(search_result.papers)} 篇候选论文："]
            for index, paper in enumerate(search_result.papers[:10], start=1):
                lines.append(
                    f"{index}. {paper.title} ({paper.year or 'N/A'}) | {paper.source} | citations={paper.citations}"
                )
            answer = "\n".join(lines)
        else:
            materials = self._compose_materials(search_result, analyses, debate)
            prompt = self.templates.render(
                "survey_writer",
                topic=query,
                materials=materials,
            )
            answer = self.llm.call(
                prompt,
                max_tokens=settings.llm_long_output_max_tokens,
                purpose="综述写作",
            )
        self.tracer.trace_step(trace_id, "write", {"intent": intent}, {"answer_preview": answer[:500]})
        return answer

    def _compose_materials(
        self,
        search_result: SearchResult | None,
        analyses: List[PaperAnalysis],
        debate: DebateResult | None,
    ) -> str:
        parts: List[str] = []
        if search_result is not None:
            parts.append("论文列表：")
            parts.extend(
                f"- {paper.title} ({paper.year or 'N/A'}, {paper.source})"
                for paper in search_result.papers[:10]
            )
        if analyses:
            parts.append("\n分析结果：")
            parts.extend(f"- {analysis.paper.title}: {analysis.summary}" for analysis in analyses)
        if debate is not None:
            parts.append("\n辩论综合：")
            parts.append(debate.synthesis)
        return "\n".join(parts)


class CoderAgent:
    def __init__(self, llm: LLMManager, templates: PromptTemplateManager, tracer: WhiteboxTracer) -> None:
        self.llm = llm
        self.templates = templates
        self.tracer = tracer

    def run(self, query: str, analyses: List[PaperAnalysis], trace_id: str) -> str:
        materials = "\n".join(
            f"- {analysis.paper.title}: {analysis.summary}"
            for analysis in analyses
        )
        prompt = self.templates.render("code_generation", topic=query, materials=materials)
        answer = self.llm.call(prompt, purpose="代码生成")
        self.tracer.trace_step(trace_id, "coder", {"query": query}, {"answer_preview": answer[:500]})
        return answer


class MultiAgentCoordinator:
    intent_flows_full = {
        "generate_survey": ["search", "analyze", "debate", "write"],
        "compare_methods": ["search", "analyze", "debate", "write"],
        "generate_code": ["search", "analyze", "coder"],
        "search_papers": ["search", "write"],
        "daily_update": ["search", "analyze", "write"],
        "analyze_paper": ["search", "analyze", "write"],
        "explain_concept": ["search", "write"],
    }

    intent_flows_fast = {
        "generate_survey": ["search", "write"],
        "compare_methods": ["search", "write"],
        "generate_code": ["coder"],
        "search_papers": ["search", "write"],
        "daily_update": ["search", "write"],
        "analyze_paper": ["search", "write"],
        "explain_concept": ["write"],
    }

    def __init__(
        self,
        llm: LLMManager,
        retriever: HybridRetriever,
        whitelist: WhitelistManager,
        reasoning: ReasoningEngine,
        templates: PromptTemplateManager,
        tracer: WhiteboxTracer,
    ) -> None:
        self.search_agent = SearchAgent(retriever, whitelist, tracer)
        self.analyze_agent = AnalyzeAgent(llm, templates, tracer)
        self.debate_agent = DebateAgent(reasoning, tracer)
        self.write_agent = WriteAgent(llm, templates, tracer)
        self.coder_agent = CoderAgent(llm, templates, tracer)

    def execute(
        self,
        query: str,
        intent: str,
        slots: Dict[str, Any],
        mode: ExecutionMode,
        trace_id: str,
        history: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        flow_map = self.intent_flows_fast if mode == ExecutionMode.FAST else self.intent_flows_full
        flow = flow_map.get(intent, ["search", "write"])

        artifacts: Dict[str, Any] = {}
        search_result: SearchResult | None = None
        analyses: List[PaperAnalysis] = []
        debate: DebateResult | None = None

        for step in flow:
            if step == "search":
                search_result = self.search_agent.run(query, intent, slots, history or [], trace_id)
                artifacts["search_result"] = search_result
            elif step == "analyze" and search_result is not None:
                analyses = self.analyze_agent.run(search_result.papers, trace_id)
                artifacts["analyses"] = analyses
            elif step == "debate":
                debate = self.debate_agent.run(query, analyses, trace_id)
                artifacts["debate"] = debate
            elif step == "write":
                artifacts["answer"] = self.write_agent.run(intent, query, search_result, analyses, debate, trace_id)
            elif step == "coder":
                artifacts["answer"] = self.coder_agent.run(query, analyses, trace_id)

        return artifacts
