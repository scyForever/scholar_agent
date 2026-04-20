from __future__ import annotations

from typing import Any, List

from config.settings import settings
from src.core.llm import LLMManager
from src.core.models import DebateResult, PaperAnalysis, ResearchPlan, SearchResult
from src.prompt_templates.manager import PromptTemplateManager
from src.whitebox.tracer import WhiteboxTracer


class WriteAgent:
    def __init__(
        self, llm: LLMManager, templates: PromptTemplateManager, tracer: WhiteboxTracer
    ) -> None:
        self.llm = llm
        self.templates = templates
        self.tracer = tracer

    def run(
        self,
        intent: str,
        query: str,
        slots: dict[str, Any] | None,
        research_plan: ResearchPlan | None,
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
            slots = dict(slots or {})
            topic = str(slots.get("topic") or slots.get("paper_title") or query)
            materials = self._compose_materials(
                research_plan,
                search_result,
                analyses,
                debate,
                slots,
            )
            template_name, purpose = self._writer_profile(intent)
            prompt = self.templates.render(
                template_name, topic=topic, materials=materials
            )
            stage_token = self.llm.bind_stage("write")
            try:
                answer = self.llm.call(
                    prompt,
                    max_tokens=settings.llm_long_output_max_tokens,
                    purpose=purpose,
                    budgeted=True,
                )
            finally:
                self.llm.reset_stage(stage_token)
        self.tracer.trace_step(
            trace_id, "write", {"intent": intent}, {"answer_preview": answer[:500]}
        )
        return answer

    def _writer_profile(self, intent: str) -> tuple[str, str]:
        profiles = {
            "generate_survey": ("survey_writer", "综述写作"),
            "compare_methods": ("compare_writer", "方法对比写作"),
            "analyze_paper": ("paper_answer_writer", "单篇论文解读"),
            "daily_update": ("daily_update_writer", "研究动态写作"),
            "explain_concept": ("concept_writer", "概念解释写作"),
        }
        return profiles.get(intent, ("survey_writer", "结构化写作"))

    def _compose_materials(
        self,
        research_plan: ResearchPlan | None,
        search_result: SearchResult | None,
        analyses: List[PaperAnalysis],
        debate: DebateResult | None,
        slots: dict[str, Any] | None = None,
    ) -> str:
        parts: List[str] = []
        constraint_block = self._writing_constraints(slots or {})
        if constraint_block:
            parts.append("写作约束：")
            parts.extend(f"- {line}" for line in constraint_block)
        if research_plan is not None:
            parts.append("研究计划：")
            parts.append(f"- 目标：{research_plan.objective}")
            parts.extend(
                f"- 任务：{task.title} | 交付：{task.deliverable or '未指定'}"
                for task in research_plan.tasks[:6]
            )
        if search_result is not None:
            local_rag = search_result.trace.get("local_rag", {})
            local_results = local_rag.get("results") or []
            if local_results:
                parts.append("本地论文片段：")
                for chunk in local_results[:5]:
                    content = self._chunk_content(chunk)
                    source_type = self._chunk_source_type(chunk)
                    parts.append(f"- [{source_type}] {content[:600]}")
            supplement = local_rag.get("supplement") or []
            if supplement:
                parts.append("\n补充网页片段：")
                for item in supplement[:3]:
                    title = str(item.get("title") or "")
                    snippet = str(item.get("snippet") or "")
                    parts.append(f"- {title}: {snippet[:300]}")
            parts.append("论文列表：")
            parts.extend(
                f"- {paper.title} ({paper.year or 'N/A'}, {paper.source})"
                for paper in search_result.papers[:10]
            )
        if analyses:
            parts.append("\n分析结果：")
            parts.extend(
                f"- {analysis.paper.title}: {analysis.summary}" for analysis in analyses
            )
        if debate is not None:
            parts.append("\n辩论综合：")
            parts.append(debate.synthesis)
        return "\n".join(parts)

    def _writing_constraints(self, slots: dict[str, Any]) -> List[str]:
        constraints: List[str] = []
        language_map = {
            "zh": "输出语言使用中文。",
            "en": "输出语言使用英文。",
            "bilingual": "输出语言使用中英双语。",
        }
        outline_map = {
            "deep": "写作深度要求：详细、系统、尽量全面。",
            "brief": "写作深度要求：简洁概览、控制篇幅。",
        }
        organization_map = {
            "timeline": "正文组织方式：按时间线或发展脉络展开。",
            "topic": "正文组织方式：按主题线或研究问题展开。",
            "method": "正文组织方式：按方法线或技术路线展开。",
            "application": "正文组织方式：按应用场景展开。",
        }
        citation_style_map = {
            "apa": "参考文献格式使用 APA。",
            "mla": "参考文献格式使用 MLA。",
            "ieee": "参考文献格式使用 IEEE。",
            "chicago": "参考文献格式使用 Chicago。",
            "gb_t_7714": "参考文献格式使用 GB/T 7714。",
        }

        min_references = int(slots.get("min_references") or 0)
        if min_references > 0:
            constraints.append(f"参考文献数量要求：不少于 {min_references} 篇。")

        language = str(slots.get("language") or "")
        if language in language_map:
            constraints.append(language_map[language])

        outline_depth = str(slots.get("outline_depth") or "")
        if outline_depth in outline_map:
            constraints.append(outline_map[outline_depth])

        organization_style = str(slots.get("organization_style") or "")
        if organization_style in organization_map:
            constraints.append(organization_map[organization_style])

        required_sections = slots.get("required_sections") or []
        if required_sections:
            constraints.append("必须包含以下章节或部分：" + "、".join(str(item) for item in required_sections if str(item).strip()) + "。")

        citation_style = str(slots.get("citation_style") or "")
        if citation_style in citation_style_map:
            constraints.append(citation_style_map[citation_style])

        return constraints

    def _chunk_content(self, chunk: Any) -> str:
        if hasattr(chunk, "content"):
            return str(chunk.content or "")
        if isinstance(chunk, dict):
            return str(chunk.get("content") or "")
        return str(chunk or "")

    def _chunk_source_type(self, chunk: Any) -> str:
        if hasattr(chunk, "source_type"):
            return str(chunk.source_type or "chunk")
        if isinstance(chunk, dict):
            return str(chunk.get("source_type") or "chunk")
        return "chunk"
