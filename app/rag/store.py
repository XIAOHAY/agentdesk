"""最小可用向量库（内存 + JSON 持久化）。接口与 Qdrant/PGVector 对齐，便于替换。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List

import numpy as np


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    embedding: List[float]


class VectorStore:
    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._matrix = None

    def add(self, chunks: List[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._matrix = None

    def _ensure_matrix(self) -> None:
        if self._matrix is None and self._chunks:
            self._matrix = np.array([c.embedding for c in self._chunks], dtype=np.float32)

    def search(self, query_vec: List[float], top_k: int = 5):
        self._ensure_matrix()
        if not self._chunks or self._matrix is None:
            return []
        q = np.array(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        mn = np.linalg.norm(self._matrix, axis=1)
        mn[mn == 0] = 1.0
        scores = (self._matrix @ q) / (mn * qn)
        idx = np.argsort(-scores)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in idx]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self._chunks], f, ensure_ascii=False)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._chunks = [Chunk(**d) for d in data]
        self._matrix = None

    @property
    def chunks(self) -> List[Chunk]:
        return self._chunks

    def __len__(self) -> int:
        return len(self._chunks)
