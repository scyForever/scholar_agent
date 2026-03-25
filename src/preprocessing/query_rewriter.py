from __future__ import annotations

from typing import List


TOPIC_KEYWORDS = {
    "强化学习": "reinforcement learning",
    "深度学习": "deep learning",
    "多智能体": "multi-agent",
    "多智能体强化学习": "multi-agent reinforcement learning",
    "大语言模型": "large language model",
    "检索增强生成": "retrieval augmented generation",
    "知识图谱": "knowledge graph",
    "时间序列": "time series",
    "图神经网络": "graph neural network",
    "因果推断": "causal inference",
}


class QueryRewriter:
    def normalize_topic(self, topic: str) -> str:
        normalized = topic.strip()
        for zh, en in TOPIC_KEYWORDS.items():
            if zh in normalized and en not in normalized.lower():
                normalized = normalized.replace(zh, f"{zh} {en}")
        return normalized

    def rewrite(self, query: str, intent: str = "search_papers") -> List[str]:
        normalized = self.normalize_topic(query)
        variants = [normalized]

        if normalized != query:
            variants.append(query)

        if intent in {"search_papers", "generate_survey", "daily_update"}:
            variants.append(f"{normalized} survey review")
            variants.append(f"{normalized} recent advances")

        if intent == "compare_methods":
            variants.append(f"{normalized} comparison benchmark")

        deduped: List[str] = []
        for item in variants:
            item = item.strip()
            if item and item not in deduped:
                deduped.append(item)
        return deduped
