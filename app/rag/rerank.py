"""Rerank（重排）。

生产建议用 cross-encoder（bge-reranker-v2）做 query-doc 相关性打分。
脚手架默认用轻量词项重叠分（无依赖、可离线），保持接口一致，
后续替换为 bge-reranker 只需改 score_pairs()。
"""
from __future__ import annotations

from typing import List

from app.rag.store import Chunk
from app.rag.tokenize import tokenize


def _overlap_score(query: str, text: str) -> float:
    q = set(tokenize(query))
    d = set(tokenize(text))
    if not q:
        return 0.0
    return len(q & d) / len(q)


def rerank(query: str, candidates: List[Chunk], top_k: int = 5) -> List[tuple[Chunk, float]]:
    # TODO 升级：用 bge-reranker-v2 的 cross-encoder 分数替换 _overlap_score
    scored = [(c, _overlap_score(query, c.text)) for c in candidates]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]
