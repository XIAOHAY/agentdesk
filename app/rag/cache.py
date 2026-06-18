"""Embedding 缓存：优先 Redis，不可用时进程内 dict 回退。

降低重复检索/重复文本的 embedding 计算与 API 调用开销（面试常问的成本/延迟优化）。
key = md5(model + text)，value = JSON 序列化的向量。
"""
from __future__ import annotations

import hashlib
import json
from typing import List, Optional

from app.config import settings


class EmbeddingCache:
    def __init__(self) -> None:
        self._mem: dict[str, List[float]] = {}
        self._redis = None
        url = getattr(settings, "redis_url", "")
        if url:
            try:
                import redis  # 可选依赖
                self._redis = redis.from_url(url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None  # 连接不上就退化为内存

    @staticmethod
    def _key(text: str) -> str:
        raw = f"{settings.embedding_model}:{text}"
        return "emb:" + hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        k = self._key(text)
        if self._redis is not None:
            v = self._redis.get(k)
            return json.loads(v) if v else None
        return self._mem.get(k)

    def set(self, text: str, vec: List[float]) -> None:
        k = self._key(text)
        if self._redis is not None:
            self._redis.set(k, json.dumps(vec), ex=3600)
        else:
            self._mem[k] = vec

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"


cache = EmbeddingCache()
