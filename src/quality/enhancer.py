from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from src.core.llm import LLMManager
from src.core.models import MoAResult, VerificationResult


class QualityEnhancer:
    def __init__(self, llm: LLMManager | None = None) -> None:
        self.llm = llm or LLMManager()

    def self_moa(self, query: str, context: str) -> MoAResult:
        real_providers = [name for name in self.llm.providers if name != "mock"]
        explicit_providers = [name for name in real_providers if name != "scnet"][:3]
        candidates: List[str] = []

        if explicit_providers:
            for provider in explicit_providers:
                candidates.append(
                    self.llm.call(
                        f"问题：{query}\n上下文：{context}\n请给出你的最佳回答。",
                        provider=provider,
                        max_tokens=settings.llm_long_output_max_tokens,
                        purpose=f"质量增强候选-{provider}",
                    )
                )
        elif real_providers:
            for temperature in (0.1, 0.4, 0.7):
                candidates.append(
                    self.llm.call(
                        f"问题：{query}\n上下文：{context}\n请给出你的最佳回答。",
                        temperature=temperature,
                        max_tokens=settings.llm_long_output_max_tokens,
                        purpose="质量增强候选",
                    )
                )
        else:
            for temperature in (0.1, 0.4, 0.7):
                candidates.append(
                    self.llm.call(
                        f"问题：{query}\n上下文：{context}\n请给出你的最佳回答。",
                        temperature=temperature,
                        max_tokens=settings.llm_long_output_max_tokens,
                        purpose="质量增强候选",
                    )
                )

        aggregate_prompt = (
            "你将收到多个候选答案，请聚合其共同结论，保留差异点，并输出更强的最终答案。\n"
            f"问题：{query}\n候选答案：\n" + "\n\n".join(candidates)
        )
        answer = self.llm.call(
            aggregate_prompt,
            max_tokens=settings.llm_long_output_max_tokens,
            purpose="质量增强聚合",
        )
        return MoAResult(answer=answer, candidates=candidates, rationale="Aggregated candidate responses.", score=0.82)

    def mpsc_verify(self, query: str, answer: str) -> VerificationResult:
        paths = [
            self.llm.call(f"请从理论角度验证以下回答是否成立。\n问题：{query}\n回答：{answer}", purpose="质量校验-理论"),
            self.llm.call(f"请从实验与证据角度验证以下回答是否成立。\n问题：{query}\n回答：{answer}", purpose="质量校验-证据"),
            self.llm.call(f"请从限制条件与边界情况角度验证以下回答是否成立。\n问题：{query}\n回答：{answer}", purpose="质量校验-边界"),
        ]

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(paths)
        sim = cosine_similarity(matrix)
        consistency = float((sim.sum() - len(paths)) / (len(paths) * (len(paths) - 1)))

        if consistency >= 0.7:
            verdict = "high_consistency"
        elif consistency >= 0.45:
            verdict = "medium_consistency"
        else:
            verdict = "low_consistency"

        suggestions = []
        if verdict != "high_consistency":
            suggestions.append("建议补充证据或缩小结论范围。")

        return VerificationResult(
            answer=answer,
            consistency_score=consistency,
            paths=paths,
            verdict=verdict,
            suggestions=suggestions,
        )
