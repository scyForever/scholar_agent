from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

from src.core.models import Paper, PaperAsset, ParsedDocument, ResearchPlan, ResearchTask
from src.memory.manager import MemoryManager
from src.tools.research_document_tool import DOCUMENT_SERVICE, fetch_paper
from src.tools.research_search_tool import search_platforms


YEAR_SPAN_RE = re.compile(r"(20\d{2})\s*[-到至]\s*(20\d{2})")
RECENT_YEARS_RE = re.compile(r"(近|最近)\s*(\d+)\s*年")

CITATION_STYLE_LABELS = {
    "apa": "APA",
    "mla": "MLA",
    "ieee": "IEEE",
    "chicago": "Chicago",
    "gb_t_7714": "GB/T 7714",
}


class ResearchMemorySkill:
    def __init__(self, memory_manager: MemoryManager | None = None) -> None:
        self.memory = memory_manager or MemoryManager()

    def remember_preference(self, user_id: str, preference: str, *, metadata: Dict[str, Any] | None = None) -> str:
        return self.memory.remember_preference(user_id, preference, metadata=metadata)

    def remember_paper(
        self,
        user_id: str,
        paper: Paper,
        summary: str,
        *,
        highlights: List[str] | None = None,
    ) -> str:
        return self.memory.remember_paper(user_id, paper, summary, highlights=highlights)

    def recall_context(self, user_id: str, query: str, *, limit: int = 8) -> List[Dict[str, Any]]:
        records = self.memory.recall_research_context(user_id, query, limit=limit)
        return [
            {
                "id": record.memory_id,
                "type": record.memory_type.value,
                "content": record.content,
                "score": record.score,
                "keywords": record.metadata.get("keywords", []),
                "metadata": record.metadata,
            }
            for record in records
        ]

    def rank_unseen_first(
        self,
        user_id: str,
        papers: Sequence[Paper],
        *,
        limit: int,
    ) -> Dict[str, Any]:
        known_keys = self.memory.seen_paper_keys(user_id)
        unseen: List[Paper] = []
        seen: List[Paper] = []
        for paper in papers:
            if self._paper_seen(paper, known_keys):
                seen.append(paper)
            else:
                unseen.append(paper)
        ranked = [*unseen, *seen][:limit]
        return {
            "papers": ranked,
            "seen_count": len(seen),
            "unseen_count": len(unseen),
            "seen_titles": [paper.title for paper in seen[:10]],
        }

    def remember_search_preferences(
        self,
        user_id: str,
        *,
        topic: str,
        time_range: str = "",
        sources: Sequence[str] | None = None,
        max_results: int | None = None,
    ) -> None:
        parts = [f"研究主题偏好：{topic}"]
        if time_range:
            parts.append(f"时间范围偏好：{time_range}")
        if sources:
            parts.append("来源偏好：" + "、".join(item for item in sources if item))
        if max_results:
            parts.append(f"检索规模偏好：{max_results} 篇")
        self.remember_preference(
            user_id,
            "\n".join(parts),
            metadata={
                "topic": topic,
                "time_range": time_range,
                "sources": list(sources or []),
                "max_results": max_results or 0,
            },
        )

    def _paper_seen(self, paper: Paper, known_keys: Iterable[str]) -> bool:
        known = set(known_keys)
        candidates = {
            (paper.paper_id or "").strip().lower(),
            (paper.title or "").strip().lower(),
            (paper.doi or "").strip().lower(),
            (paper.arxiv_id or "").strip().lower(),
            (paper.pmid or "").strip().lower(),
        }
        candidates = {item for item in candidates if item}
        return bool(candidates.intersection(known))


class LiteratureSearchSkill:
    def __init__(self, memory_skill: ResearchMemorySkill | None = None) -> None:
        self.memory_skill = memory_skill

    def search(
        self,
        query: str,
        *,
        platforms: Sequence[str] | None = None,
        max_results: int = 10,
        time_range: str = "",
        author: str = "",
        user_id: str = "",
    ) -> Dict[str, Any]:
        result = search_platforms(
            query,
            platforms=platforms,
            max_results=max_results,
            time_range=time_range,
            author=author,
        )
        papers: List[Paper] = list(result.get("papers") or [])
        memory_trace: Dict[str, Any] = {}
        if user_id and self.memory_skill is not None:
            ranked = self.memory_skill.rank_unseen_first(user_id, papers, limit=max_results)
            papers = list(ranked["papers"])
            memory_trace = {
                "seen_count": ranked["seen_count"],
                "unseen_count": ranked["unseen_count"],
                "seen_titles": ranked["seen_titles"],
            }
            self.memory_skill.remember_search_preferences(
                user_id,
                topic=query,
                time_range=time_range,
                sources=result.get("platforms") or [],
                max_results=max_results,
            )
        return {
            **result,
            "papers": papers[:max_results],
            "memory_trace": memory_trace,
        }


class DeepReadingSkill:
    def fetch_full_text(
        self,
        identifier: str,
        *,
        identifier_type: str = "auto",
        prefer: str = "pdf",
        download_dir: str = "",
    ) -> PaperAsset:
        return fetch_paper(
            identifier,
            identifier_type=identifier_type,
            prefer=prefer,
            download_dir=download_dir,
        )

    def parse_pdf(self, pdf_path: str, *, target_section: str = "") -> ParsedDocument:
        return DOCUMENT_SERVICE.parse_pdf(pdf_path, target_section=target_section)

    def targeted_read(
        self,
        pdf_path: str,
        *,
        section_name: str,
        max_chunks: int = 5,
    ) -> Dict[str, Any]:
        return DOCUMENT_SERVICE.read_section(pdf_path, section_name=section_name, max_chunks=max_chunks)

    def extract_visuals(
        self,
        pdf_path: str,
        *,
        page_numbers: List[int] | None = None,
        output_dir: str = "",
    ) -> Dict[str, Any]:
        return DOCUMENT_SERVICE.extract_visuals(pdf_path, page_numbers=page_numbers, output_dir=output_dir)


class ResearchPlanningSkill:
    def plan(self, topic: str, *, intent: str = "generate_survey", slots: Dict[str, Any] | None = None) -> ResearchPlan:
        slots = dict(slots or {})
        time_range = str(slots.get("time_range") or self._extract_time_range(topic))
        platforms = self._suggest_platforms(topic)
        constraints = self._writing_constraints(slots)
        objective = self._objective(topic, intent=intent, time_range=time_range, constraints=constraints)
        tasks = [
            ResearchTask(
                task_id="task-1-scope",
                title="界定研究范围",
                goal=self._scope_goal(constraints),
                deliverable="关键词表、写作约束表与检索式草稿",
                suggested_tools=["search_literature"],
                metadata={"time_range": time_range, "constraints": constraints},
            ),
            ResearchTask(
                task_id="task-2-search",
                title="执行多源检索",
                goal=self._search_goal(constraints),
                deliverable=self._search_deliverable(constraints),
                dependencies=["task-1-scope"],
                suggested_tools=platforms,
                metadata={"time_range": time_range, "constraints": constraints},
            ),
            ResearchTask(
                task_id="task-3-screen",
                title="初筛与聚类",
                goal=self._screen_goal(constraints),
                deliverable=self._screen_deliverable(constraints),
                dependencies=["task-2-search"],
                suggested_tools=["search_literature"],
                metadata={"constraints": constraints},
            ),
            ResearchTask(
                task_id="task-4-read",
                title="精读核心论文",
                goal=self._reading_goal(constraints),
                deliverable=self._reading_deliverable(constraints),
                dependencies=["task-3-screen"],
                suggested_tools=["fetch_paper_asset", "parse_pdf_document", "extract_paper_visuals", "read_paper_section"],
                metadata={"constraints": constraints},
            ),
            ResearchTask(
                task_id="task-5-synthesize",
                title="提炼共识与分歧",
                goal=self._synthesis_goal(constraints),
                deliverable=self._synthesis_deliverable(constraints),
                dependencies=["task-4-read"],
                suggested_tools=["read_paper_section"],
                metadata={"constraints": constraints},
            ),
            ResearchTask(
                task_id="task-6-write",
                title="输出最终交付",
                goal=self._write_goal(constraints),
                deliverable=self._write_deliverable(constraints),
                dependencies=["task-5-synthesize"],
                suggested_tools=["search_literature"],
                metadata={"constraints": constraints},
            ),
        ]
        return ResearchPlan(
            topic=topic,
            objective=objective,
            tasks=tasks,
            milestones=[
                "完成检索式、写作约束和纳入标准",
                self._search_milestone(constraints),
                "完成定向精读与证据抽取",
                self._write_milestone(constraints),
            ],
            validation=self._validation_items(constraints),
            risks=self._risk_items(constraints),
            metadata={
                "intent": intent,
                "time_range": time_range,
                "platforms": platforms,
                "writing_constraints": constraints,
            },
        )

    def _extract_time_range(self, topic: str) -> str:
        match = YEAR_SPAN_RE.search(topic)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        recent = RECENT_YEARS_RE.search(topic)
        if recent:
            return f"last_{recent.group(2)}_years"
        return ""

    def _suggest_platforms(self, topic: str) -> List[str]:
        lowered = topic.lower()
        base = ["search_arxiv", "search_openalex", "search_semantic_scholar"]
        if any(token in lowered for token in ("医学", "生物", "clinical", "biomedical", "drug", "patient")):
            base.append("search_pubmed")
        if any(token in lowered for token in ("电气", "通信", "机器人", "芯片", "signal", "wireless", "ieee")):
            base.append("search_ieee_xplore")
        base.append("search_google_scholar")
        return list(dict.fromkeys(base))

    def _objective(self, topic: str, *, intent: str, time_range: str, constraints: Dict[str, Any]) -> str:
        suffix = f"，重点覆盖 {time_range}" if time_range else ""
        constraint_suffix = self._constraint_objective_suffix(constraints)
        if intent == "generate_survey":
            return f"围绕“{topic}”形成一份可溯源综述{suffix}{constraint_suffix}。"
        if intent == "compare_methods":
            return f"围绕“{topic}”形成方法对比与证据归纳{suffix}{constraint_suffix}。"
        return f"围绕“{topic}”形成结构化研究执行方案{suffix}{constraint_suffix}。"

    def _writing_constraints(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "min_references": int(slots.get("min_references") or 0),
            "language": str(slots.get("language") or ""),
            "outline_depth": str(slots.get("outline_depth") or ""),
            "organization_style": str(slots.get("organization_style") or ""),
            "required_sections": [str(item) for item in (slots.get("required_sections") or []) if str(item).strip()],
            "citation_style": str(slots.get("citation_style") or ""),
        }

    def _constraint_objective_suffix(self, constraints: Dict[str, Any]) -> str:
        parts: List[str] = []
        language = str(constraints.get("language") or "")
        if language == "zh":
            parts.append("中文输出")
        elif language == "en":
            parts.append("英文输出")
        elif language == "bilingual":
            parts.append("中英双语输出")

        organization_style = str(constraints.get("organization_style") or "")
        if organization_style == "timeline":
            parts.append("按时间线组织")
        elif organization_style == "topic":
            parts.append("按主题线组织")
        elif organization_style == "method":
            parts.append("按方法线组织")
        elif organization_style == "application":
            parts.append("按应用线组织")

        min_references = int(constraints.get("min_references") or 0)
        if min_references > 0:
            parts.append(f"参考文献不少于 {min_references} 篇")
        return "，并满足" + "、".join(parts) if parts else ""

    def _scope_goal(self, constraints: Dict[str, Any]) -> str:
        goal = "明确主题边界、关键词、纳入排除标准，并梳理写作约束。"
        if constraints.get("required_sections"):
            goal += " 先把必含章节映射到证据收集清单。"
        return goal

    def _search_goal(self, constraints: Dict[str, Any]) -> str:
        goal = "按平台分批检索候选文献并保留来源信息。"
        min_references = int(constraints.get("min_references") or 0)
        if min_references > 0:
            goal += f" 候选池规模至少覆盖 {min_references} 篇可引用文献。"
        return goal

    def _search_deliverable(self, constraints: Dict[str, Any]) -> str:
        min_references = int(constraints.get("min_references") or 0)
        if min_references > 0:
            return f"候选论文池、元数据表与不少于 {min_references} 篇的初始参考文献清单"
        return "候选论文池与元数据表"

    def _screen_goal(self, constraints: Dict[str, Any]) -> str:
        organization_style = str(constraints.get("organization_style") or "")
        if organization_style == "timeline":
            return "剔除重复和离题论文，并按时间线聚类核心研究阶段。"
        if organization_style == "topic":
            return "剔除重复和离题论文，并按主题线或研究问题聚类。"
        if organization_style == "method":
            return "剔除重复和离题论文，并按方法路线聚类。"
        if organization_style == "application":
            return "剔除重复和离题论文，并按应用场景聚类。"
        return "剔除重复和离题论文，按方法、任务或数据集聚类。"

    def _screen_deliverable(self, constraints: Dict[str, Any]) -> str:
        if str(constraints.get("organization_style") or "") == "timeline":
            return "去重后的核心论文列表与时间线分组结果"
        return "去重后的核心论文列表"

    def _reading_goal(self, constraints: Dict[str, Any]) -> str:
        goal = "重点读取 Methodology、Experiments、Conclusion 等章节，并抽取图表与公式。"
        if constraints.get("required_sections"):
            goal += " 对用户要求的章节优先补齐证据。"
        return goal

    def _reading_deliverable(self, constraints: Dict[str, Any]) -> str:
        if constraints.get("required_sections"):
            return "结构化阅读笔记与必含章节证据摘录"
        return "结构化阅读笔记"

    def _synthesis_goal(self, constraints: Dict[str, Any]) -> str:
        organization_style = str(constraints.get("organization_style") or "")
        if organization_style == "timeline":
            return "沿时间线总结核心方案、证据强度、局限和开放问题。"
        if organization_style == "topic":
            return "围绕主题主线总结核心方案、证据强度、局限和开放问题。"
        if organization_style == "method":
            return "沿方法路线总结核心方案、证据强度、局限和开放问题。"
        if organization_style == "application":
            return "沿应用场景总结核心方案、证据强度、局限和开放问题。"
        return "总结核心方案、证据强度、局限和开放问题。"

    def _synthesis_deliverable(self, constraints: Dict[str, Any]) -> str:
        if str(constraints.get("organization_style") or "") == "timeline":
            return "时间线式综述框架或阶段划分矩阵"
        return "主题综述框架或对比矩阵"

    def _write_goal(self, constraints: Dict[str, Any]) -> str:
        goal = "生成综述、提案、草稿或研究日报，并保留引用链路。"
        if constraints.get("citation_style"):
            goal += " 输出时遵循指定参考文献格式。"
        if constraints.get("language"):
            goal += " 使用指定语言完成成稿。"
        return goal

    def _write_deliverable(self, constraints: Dict[str, Any]) -> str:
        deliverable = "最终文稿与参考文献列表"
        extras: List[str] = []
        if constraints.get("citation_style"):
            extras.append(f"{self._citation_style_label(constraints)} 引用格式")
        if constraints.get("required_sections"):
            extras.append("包含指定章节")
        if extras:
            deliverable += "（" + "、".join(extras) + "）"
        return deliverable

    def _search_milestone(self, constraints: Dict[str, Any]) -> str:
        min_references = int(constraints.get("min_references") or 0)
        if min_references > 0:
            return f"形成不少于 {min_references} 篇的候选文献池"
        return "形成核心论文池"

    def _write_milestone(self, constraints: Dict[str, Any]) -> str:
        if constraints.get("required_sections"):
            return "完成提纲、指定章节和初稿"
        return "完成提纲和初稿"

    def _validation_items(self, constraints: Dict[str, Any]) -> List[str]:
        items = [
            "检索平台、年份、关键词和去重规则可复现",
            "每个核心结论至少能回溯到 1 篇原始论文",
            "方法、实验和局限部分都包含章节级证据",
        ]
        min_references = int(constraints.get("min_references") or 0)
        if min_references > 0:
            items.append(f"最终参考文献数量不少于 {min_references} 篇")
        if constraints.get("required_sections"):
            items.append("最终文稿完整包含指定章节：" + "、".join(constraints["required_sections"]))
        if constraints.get("citation_style"):
            items.append(f"参考文献格式符合 {self._citation_style_label(constraints)} 要求")
        return items

    def _risk_items(self, constraints: Dict[str, Any]) -> List[str]:
        items = [
            "Google Scholar 与 IEEE Xplore 可能受配额或接口限制",
            "仅靠摘要筛选容易漏掉真实方法差异",
            "PDF 双栏和扫描版文档可能需要 OCR 补强",
        ]
        min_references = int(constraints.get("min_references") or 0)
        if min_references > 0:
            items.append("若主题过新或过窄，满足最低参考文献数量可能需要扩大来源与检索式")
        if str(constraints.get("organization_style") or "") == "timeline":
            items.append("按时间线组织时，若早期论文元数据缺失，阶段划分可能不稳定")
        return items

    def _citation_style_label(self, constraints: Dict[str, Any]) -> str:
        style = str(constraints.get("citation_style") or "").strip()
        return CITATION_STYLE_LABELS.get(style, style or "指定")


class ResearchSkillset:
    def __init__(self, memory_manager: MemoryManager | None = None) -> None:
        self.memory = ResearchMemorySkill(memory_manager)
        self.search = LiteratureSearchSkill(memory_skill=self.memory)
        self.reading = DeepReadingSkill()
        self.planning = ResearchPlanningSkill()
