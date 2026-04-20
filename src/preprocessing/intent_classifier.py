from __future__ import annotations

from typing import Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.intent_config import INTENT_SPECS
from src.core.llm import LLMManager
from src.core.structured_outputs import IntentClassificationOutput
from src.prompt_templates.manager import PromptTemplateManager


class IntentClassifier:
    RULE_PATTERNS = [
        ("search_papers", ("搜索", "检索", "找论文", "查论文", "找文献", "查文献")),
        ("generate_survey", ("综述", "survey", "研究现状", "写一篇")),
        ("compare_methods", ("比较", "对比", "区别", "vs", "compare")),
        ("analyze_paper", ("分析这篇论文", "解读论文", "这篇论文", "paper title", "pdf")),
        ("generate_code", ("代码", "实现", "伪代码", "复现")),
        ("daily_update", ("最近进展", "最新进展", "每日更新", "daily update")),
        ("explain_concept", ("解释", "是什么", "什么意思", "介绍一下", "原理")),
    ]
    LEXICAL_SHORT_CIRCUIT = 0.18

    def __init__(
        self,
        llm: LLMManager | None = None,
        template_manager: PromptTemplateManager | None = None,
    ) -> None:
        self.llm = llm or LLMManager()
        self.templates = template_manager or PromptTemplateManager()

    def classify(self, query: str) -> Dict[str, object]:
        rule_based = self._classify_by_rules(query)
        if rule_based is not None:
            return rule_based

        lexical_intent, lexical_score = self._classify_lexically(query)
        if lexical_score >= self.LEXICAL_SHORT_CIRCUIT or not self.llm.has_verified_provider():
            return {
                "intent": lexical_intent,
                "confidence": lexical_score,
                "reason": "规则/Tf-idf 本地快速意图识别。",
            }

        llm_result = self._classify_with_llm(query)

        if llm_result.get("intent") in INTENT_SPECS and float(llm_result.get("confidence", 0.0)) >= lexical_score:
            return llm_result
        return {
            "intent": lexical_intent,
            "confidence": lexical_score,
            "reason": "TF-IDF similarity over intent descriptions.",
        }

    def _classify_by_rules(self, query: str) -> Dict[str, object] | None:
        normalized = query.strip().lower()
        for intent, markers in self.RULE_PATTERNS:
            if any(marker.lower() in normalized for marker in markers):
                return {
                    "intent": intent,
                    "confidence": 0.92,
                    "reason": "规则命中，跳过远程 LLM 意图识别。",
                }
        return None

    def _classify_lexically(self, query: str) -> Tuple[str, float]:
        labels = list(INTENT_SPECS.keys())
        descriptions = [INTENT_SPECS[label].description for label in labels]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(descriptions + [query])
        scores = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
        best_idx = int(scores.argmax())
        return labels[best_idx], float(scores[best_idx])

    def _classify_with_llm(self, query: str) -> Dict[str, object]:
        prompt = self.templates.render(
            "intent_classification",
            intent_options=", ".join(INTENT_SPECS),
            query=query,
        )
        result = self.llm.call_structured(
            prompt,
            IntentClassificationOutput,
            purpose="意图识别",
        )
        return {
            "intent": result.intent,
            "confidence": float(result.confidence),
            "reason": result.reason,
        }
