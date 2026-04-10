from __future__ import annotations

import re
from typing import Any, Dict, List

from src.core.models import ResearchPlan, ResearchTask


YEAR_SPAN_RE = re.compile(r"(20\d{2})\s*[-到至]\s*(20\d{2})")
RECENT_YEARS_RE = re.compile(r"(近|最近)\s*(\d+)\s*年")


class ResearchPlanningComponent:
    def plan(self, topic: str, *, intent: str = "generate_survey", slots: Dict[str, Any] | None = None) -> ResearchPlan:
        slots = dict(slots or {})
        time_range = str(slots.get("time_range") or self._extract_time_range(topic))
        platforms = self._suggest_platforms(topic)
        objective = self._objective(topic, intent=intent, time_range=time_range)
        tasks = [
            ResearchTask(
                task_id="task-1-scope",
                title="界定研究范围",
                goal="明确主题边界、关键词、纳入排除标准。",
                deliverable="关键词表与检索式草稿",
                suggested_tools=["search_literature"],
                metadata={"time_range": time_range},
            ),
            ResearchTask(
                task_id="task-2-search",
                title="执行多源检索",
                goal="按平台分批检索候选文献并保留来源信息。",
                deliverable="候选论文池与元数据表",
                dependencies=["task-1-scope"],
                suggested_tools=platforms,
                metadata={"time_range": time_range},
            ),
            ResearchTask(
                task_id="task-3-screen",
                title="初筛与聚类",
                goal="剔除重复和离题论文，按方法、任务或数据集聚类。",
                deliverable="去重后的核心论文列表",
                dependencies=["task-2-search"],
                suggested_tools=["search_literature"],
            ),
            ResearchTask(
                task_id="task-4-read",
                title="精读核心论文",
                goal="重点读取 Methodology、Experiments、Conclusion 等章节，并抽取图表与公式。",
                deliverable="结构化阅读笔记",
                dependencies=["task-3-screen"],
                suggested_tools=["fetch_paper_asset", "parse_pdf_document", "extract_paper_visuals", "read_paper_section"],
            ),
            ResearchTask(
                task_id="task-5-synthesize",
                title="提炼共识与分歧",
                goal="总结核心方案、证据强度、局限和开放问题。",
                deliverable="主题综述框架或对比矩阵",
                dependencies=["task-4-read"],
                suggested_tools=["read_paper_section"],
            ),
            ResearchTask(
                task_id="task-6-write",
                title="输出最终交付",
                goal="生成综述、提案、草稿或研究日报，并保留引用链路。",
                deliverable="最终文稿与参考文献列表",
                dependencies=["task-5-synthesize"],
                suggested_tools=["search_literature"],
            ),
        ]
        return ResearchPlan(
            topic=topic,
            objective=objective,
            tasks=tasks,
            milestones=[
                "完成检索式与纳入标准",
                "形成核心论文池",
                "完成定向精读与证据抽取",
                "完成提纲和初稿",
            ],
            validation=[
                "检索平台、年份、关键词和去重规则可复现",
                "每个核心结论至少能回溯到 1 篇原始论文",
                "方法、实验和局限部分都包含章节级证据",
            ],
            risks=[
                "Google Scholar 与 IEEE Xplore 可能受配额或接口限制",
                "仅靠摘要筛选容易漏掉真实方法差异",
                "PDF 双栏和扫描版文档可能需要 OCR 补强",
            ],
            metadata={
                "intent": intent,
                "time_range": time_range,
                "platforms": platforms,
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

    def _objective(self, topic: str, *, intent: str, time_range: str) -> str:
        suffix = f"，重点覆盖 {time_range}" if time_range else ""
        if intent == "generate_survey":
            return f"围绕“{topic}”形成一份可溯源综述{suffix}。"
        if intent == "compare_methods":
            return f"围绕“{topic}”形成方法对比与证据归纳{suffix}。"
        return f"围绕“{topic}”形成结构化研究执行方案{suffix}。"
