from __future__ import annotations

from typing import Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.intent_config import INTENT_SPECS
from src.core.llm import LLMManager
from src.core.structured_outputs import IntentClassificationOutput
from src.prompt_templates.manager import PromptTemplateManager


class IntentClassifier:
    def __init__(
        self,
        llm: LLMManager | None = None,
        template_manager: PromptTemplateManager | None = None,
    ) -> None:
        self.llm = llm or LLMManager()
        self.templates = template_manager or PromptTemplateManager()

    def classify(self, query: str) -> Dict[str, object]:
        lexical_intent, lexical_score = self._classify_lexically(query)
        real_provider_count = len([name for name in self.llm.providers if name != "mock"])
        if real_provider_count == 0:
            return {
                "intent": lexical_intent,
                "confidence": lexical_score,
                "reason": "TF-IDF similarity over intent descriptions.",
            }

        llm_result = self._classify_with_llm(query)

        if llm_result.get("intent") in INTENT_SPECS and float(llm_result.get("confidence", 0.0)) >= lexical_score:
            return llm_result
        return {
            "intent": lexical_intent,
            "confidence": lexical_score,
            "reason": "TF-IDF similarity over intent descriptions.",
        }

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
