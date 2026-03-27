from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class TaskLevel(str, Enum):
    L1_SIMPLE = "simple"
    L2_MODERATE = "moderate"
    L3_COMPLEX = "complex"
    L4_ADVANCED = "advanced"
    L5_EXPERT = "expert"


class LLMTier(str, Enum):
    LITE = "lite"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass(slots=True)
class TaskConfig:
    llm_tier: LLMTier
    max_reasoning_depth: int
    enable_multi_agent: bool
    enable_quality_enhance: bool
    reasoning_modes: List[str]
    max_llm_calls: int
    timeout_seconds: int


TASK_CONFIGS: Dict[TaskLevel, TaskConfig] = {
    TaskLevel.L1_SIMPLE: TaskConfig(LLMTier.LITE, 1, False, False, ["direct"], 2, 20),
    TaskLevel.L2_MODERATE: TaskConfig(LLMTier.LITE, 2, False, False, ["cot"], 4, 40),
    TaskLevel.L3_COMPLEX: TaskConfig(LLMTier.STANDARD, 3, True, False, ["cot", "react"], 8, 90),
    TaskLevel.L4_ADVANCED: TaskConfig(LLMTier.STANDARD, 4, True, True, ["cot", "debate", "reflection"], 16, 180),
    TaskLevel.L5_EXPERT: TaskConfig(LLMTier.STANDARD, 5, True, True, ["cot", "debate", "reflection"], 30, 300),
}

INTENT_BASE_SCORES: Dict[str, int] = {
    "search_papers": 1,
    "explain_concept": 1,
    "analyze_paper": 2,
    "daily_update": 2,
    "compare_methods": 3,
    "generate_code": 4,
    "generate_survey": 4,
}

MULTI_OBJECTIVE_MARKERS = ("并且", "同时", "以及", "还要", "并说明", "并比较", "并分析")


class TaskHierarchyPlanner:
    def classify(self, query: str, intent: str, slots: Dict[str, object]) -> tuple[TaskLevel, TaskConfig]:
        score = INTENT_BASE_SCORES.get(intent, 2)
        normalized_query = query.strip()
        if slots.get("comparison_target"):
            score += 1
        if slots.get("time_range"):
            score += 1
        if isinstance(slots.get("max_papers"), int) and int(slots["max_papers"]) >= 20:
            score += 1
        if len(normalized_query) >= 60:
            score += 1
        if any(marker in normalized_query for marker in MULTI_OBJECTIVE_MARKERS):
            score += 1
        score = min(score, 5)

        if score <= 1:
            level = TaskLevel.L1_SIMPLE
        elif score == 2:
            level = TaskLevel.L2_MODERATE
        elif score == 3:
            level = TaskLevel.L3_COMPLEX
        elif score == 4:
            level = TaskLevel.L4_ADVANCED
        else:
            level = TaskLevel.L5_EXPERT
        return level, TASK_CONFIGS[level]
