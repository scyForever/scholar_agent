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


class TaskHierarchyPlanner:
    def classify(self, query: str, intent: str, slots: Dict[str, object]) -> tuple[TaskLevel, TaskConfig]:
        score = 0
        score += min(len(query) // 40, 2)
        score += 2 if intent in {"generate_survey", "generate_code", "compare_methods"} else 0
        score += 1 if intent in {"analyze_paper", "daily_update"} else 0
        score += 1 if slots.get("comparison_target") else 0
        score += 1 if slots.get("time_range") else 0

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
