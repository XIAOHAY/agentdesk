"""极简 BM25（纯 Python，无额外依赖）。

关键词召回，弥补向量检索对专有名词/数字的不足。
混合检索的另一路召回源。
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List

from app.rag.store import Chunk
from app.rag.tokenize import tokenize


class BM25:
    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.docs = [tokenize(c.text) for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / len(self.docs)) if self.docs else 0.0
        self.freqs = [Counter(d) for d in self.docs]
        # 文档频率
        df: Counter = Counter()
        for d in self.docs:
            for term in set(d):
                df[term] += 1
        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()
        }

    def search(self, query: str, top_k: int = 20) -> List[tuple[Chunk, float]]:
        if not self.docs:
            return []
        q_terms = tokenize(query)
        scores: List[float] = []
        for i, freq in enumerate(self.freqs):
            s = 0.0
            dl = self.doc_len[i] or 1
            for term in q_terms:
                if term not in freq:
                    continue
                idf = self.idf.get(term, 0.0)
                tf = freq[term]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (tf * (self.k1 + 1)) / denom
            scores.append(s)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.chunks[i], scores[i]) for i in ranked if scores[i] > 0]
