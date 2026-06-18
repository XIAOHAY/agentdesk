"""检索器：vector / hybrid + 可选 Rerank，支持多查询融合。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.config import settings
from app.llm import embed_query
from app.rag.bm25 import BM25
from app.rag.indexer import INDEX_PATH
from app.rag.rerank import rerank
from app.rag.store import Chunk
from app.rag.store_factory import get_store

CANDIDATE_N = 20
RRF_K = 60


@dataclass
class Evidence:
    doc_id: str
    chunk_id: str
    text: str
    score: float


class Retriever:
    def __init__(self) -> None:
        self.store = get_store()
        self.store.load(INDEX_PATH)
        self.bm25 = BM25(self.store.chunks)

    def _vector(self, query: str, n: int) -> List[Chunk]:
        qv = embed_query(query)
        return [c for c, _ in self.store.search(qv, top_k=n)]

    def _bm25(self, query: str, n: int) -> List[Chunk]:
        return [c for c, _ in self.bm25.search(query, top_k=n)]

    @staticmethod
    def _rrf(rank_lists: List[List[Chunk]]) -> List[tuple]:
        scores = {}
        by_id = {}
        for lst in rank_lists:
            for rank, c in enumerate(lst):
                by_id[c.chunk_id] = c
                scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        fused = sorted(scores.items(), key=lambda x: -x[1])
        return [(by_id[cid], s) for cid, s in fused]

    def retrieve(self, query: str, mode: str = "hybrid", use_rerank: bool = True,
                 top_k: int | None = None) -> List[Evidence]:
        return self.retrieve_multi([query], mode=mode, use_rerank=use_rerank, top_k=top_k)

    def retrieve_multi(self, queries: List[str], mode: str = "hybrid",
                       use_rerank: bool = True, top_k: int | None = None) -> List[Evidence]:
        k = top_k or settings.top_k
        rank_lists: List[List[Chunk]] = []
        for q in queries:
            rank_lists.append(self._vector(q, CANDIDATE_N))
            if mode == "hybrid":
                rank_lists.append(self._bm25(q, CANDIDATE_N))

        if mode == "vector" and len(rank_lists) == 1:
            fused = [(c, 1.0 / (RRF_K + i + 1)) for i, c in enumerate(rank_lists[0])]
        else:
            fused = self._rrf(rank_lists)

        candidates = [c for c, _ in fused][:CANDIDATE_N]

        if use_rerank:
            reranked = rerank(queries[0], candidates, top_k=k)
            return [Evidence(c.doc_id, c.chunk_id, c.text, s) for c, s in reranked]
        return [Evidence(c.doc_id, c.chunk_id, c.text, s) for c, s in fused[:k]]
