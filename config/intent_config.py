from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


SUPPORTED_INTENTS = [
    "search_papers",
    "explain_concept",
    "compare_methods",
    "generate_survey",
    "generate_code",
    "analyze_paper",
    "daily_update",
]


@dataclass(slots=True)
class IntentSpec:
    name: str
    description: str
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    output_style: str = "structured"


INTENT_SPECS: Dict[str, IntentSpec] = {
    "search_papers": IntentSpec(
        name="search_papers",
        description="检索某个主题、任务或方法相关的学术论文。",
        required_slots=["topic"],
        optional_slots=["time_range", "max_papers", "sources", "language"],
    ),
    "explain_concept": IntentSpec(
        name="explain_concept",
        description="解释某个学术概念、术语或方法。",
        required_slots=["topic"],
        optional_slots=["audience", "language", "rag_mode", "context_source"],
    ),
    "compare_methods": IntentSpec(
        name="compare_methods",
        description="比较两个或多个方法、模型或论文。",
        required_slots=["topic", "comparison_target"],
        optional_slots=["criteria", "time_range", "max_papers"],
    ),
    "generate_survey": IntentSpec(
        name="generate_survey",
        description="围绕指定主题生成论文综述或研究现状总结。",
        required_slots=["topic"],
        optional_slots=[
            "time_range",
            "max_papers",
            "min_references",
            "language",
            "outline_depth",
            "organization_style",
            "required_sections",
            "citation_style",
        ],
    ),
    "generate_code": IntentSpec(
        name="generate_code",
        description="根据论文方法或研究方向生成实现代码或伪代码。",
        required_slots=["topic"],
        optional_slots=["target_framework", "paper_title", "language"],
    ),
    "analyze_paper": IntentSpec(
        name="analyze_paper",
        description="对单篇论文进行摘要、贡献、方法、实验与不足分析。",
        required_slots=["paper_title"],
        optional_slots=["focus", "pdf_path"],
    ),
    "daily_update": IntentSpec(
        name="daily_update",
        description="汇总某个领域最近的论文更新与研究动态。",
        required_slots=["topic"],
        optional_slots=["time_range", "max_papers"],
    ),
}
