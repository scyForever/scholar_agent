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
        explicit_providers = self.llm.get_verified_provider_names()[:3]
        candidates: List[str] = []
        errors: List[str] = []
        candidate_prompt = f"问题：{query}\n上下文：{context}\n请给出你的最佳回答。"

        if explicit_providers:
            for provider in explicit_providers:
                try:
                    candidates.append(
                        self.llm.call(
                            candidate_prompt,
                            provider=provider,
                            max_tokens=settings.llm_long_output_max_tokens,
                            purpose=f"质量增强候选-{provider}",
                            budgeted=True,
                        )
                    )
                except Exception as exc:
                    errors.append(f"{provider}: {type(exc).__name__}: {exc}")
        elif real_providers and self.llm.has_verified_provider():
            for temperature in (0.1, 0.4, 0.7):
                try:
                    candidates.append(
                        self.llm.call(
                            candidate_prompt,
                            temperature=temperature,
                            max_tokens=settings.llm_long_output_max_tokens,
                            purpose="质量增强候选",
                            budgeted=True,
                        )
                    )
                except Exception as exc:
                    errors.append(f"auto@{temperature}: {type(exc).__name__}: {exc}")
        else:
            for temperature in (0.1, 0.4, 0.7):
                try:
                    candidates.append(
                        self.llm.call(
                            candidate_prompt,
                            provider="mock",
                            temperature=temperature,
                            max_tokens=settings.llm_long_output_max_tokens,
                            purpose="质量增强候选",
                            budgeted=True,
                        )
                    )
                except Exception as exc:
                    errors.append(f"mock@{temperature}: {type(exc).__name__}: {exc}")

        if not candidates:
            return MoAResult(
                answer=context,
                candidates=[],
                rationale="Quality enhancement skipped because no candidate completed successfully.",
                score=0.0,
                errors=errors,
            )

        if len(candidates) == 1:
            return MoAResult(
                answer=candidates[0],
                candidates=candidates,
                rationale="Only one quality-enhancement candidate succeeded; aggregation skipped.",
                score=0.6,
                errors=errors,
            )

        aggregate_prompt = (
            "你将收到多个候选答案，请聚合其共同结论，保留差异点，并输出更强的最终答案。\n"
            f"问题：{query}\n候选答案：\n" + "\n\n".join(candidates)
        )
        try:
            answer = self.llm.call(
                aggregate_prompt,
                max_tokens=settings.llm_long_output_max_tokens,
                purpose="质量增强聚合",
                budgeted=True,
            )
        except Exception as exc:
            errors.append(f"aggregate: {type(exc).__name__}: {exc}")
            return MoAResult(
                answer=context,
                candidates=candidates,
                rationale="Quality-enhancement aggregation failed; preserved the pre-enhancement answer.",
                score=0.0,
                errors=errors,
            )
        return MoAResult(
            answer=answer,
            candidates=candidates,
            rationale="Aggregated candidate responses.",
            score=0.82,
            errors=errors,
        )

    def mpsc_verify(self, query: str, answer: str) -> VerificationResult:
        prompts = [
            ("理论", "质量校验-理论"),
            ("证据", "质量校验-证据"),
            ("边界", "质量校验-边界"),
        ]
        paths: List[str] = []
        errors: List[str] = []
        for label, purpose in prompts:
            try:
                paths.append(
                    self.llm.call(
                        f"请从{label}角度验证以下回答是否成立。\n问题：{query}\n回答：{answer}",
                        purpose=purpose,
                        budgeted=True,
                    )
                )
            except Exception as exc:
                errors.append(f"{purpose}: {type(exc).__name__}: {exc}")

        if len(paths) < 2:
            return VerificationResult(
                answer=answer,
                consistency_score=0.0,
                paths=paths,
                verdict="skipped_due_to_llm_error",
                suggestions=["质量校验未完成，已保留当前答案。"],
                errors=errors,
            )

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
            errors=errors,
        )
