from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List


def tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"[\s,.;:!?()，。；：！？、]+", text.lower()) if token]


def bm25_scores(query: str, documents: Iterable[str]) -> List[float]:
    docs = list(documents)
    tokenized_docs = [tokenize(doc) for doc in docs]
    query_tokens = tokenize(query)
    if not docs or not query_tokens:
        return [0.0 for _ in docs]

    doc_count = len(tokenized_docs)
    avgdl = sum(len(doc) for doc in tokenized_docs) / max(doc_count, 1)
    doc_freq: Counter[str] = Counter()
    for doc in tokenized_docs:
        for token in set(doc):
            doc_freq[token] += 1

    scores: List[float] = []
    k1 = 1.5
    b = 0.75
    for doc in tokenized_docs:
        tf = Counter(doc)
        doc_len = len(doc) or 1
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
            numerator = tf[token] * (k1 + 1)
            denominator = tf[token] + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * numerator / denominator
        scores.append(score)
    return scores
