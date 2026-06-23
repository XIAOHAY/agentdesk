"""记忆专用存储：复用 QdrantStore 风格，按 user_id 做 namespace 隔离。

接口（两种实现一致）：
    upsert(rec)                 -> None
    search(user_id, vec, top_k) -> List[(MemoryRecord, score)]
    delete(user_id, mem_ids)    -> None
    list_by_user(user_id)       -> List[MemoryRecord]

后端选择与 rag.store_factory 同构：vector_backend=qdrant 且可连 → Qdrant；
否则回退进程内 InMemoryMemoryStore。记忆向量库独立 collection（settings.mem_collection），
不与 RAG 知识库混用，避免“知识”与“记忆”互相污染检索。
"""
from __future__ import annotations

import uuid
from typing import List, Tuple

import numpy as np

from app.config import settings
from app.memory.schema import MemoryRecord


def _cosine(a: List[float], b: List[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


class InMemoryMemoryStore:
    """进程内记忆库（dict + numpy 余弦）。重启即丢，仅用于无 Qdrant 的演示。"""

    def __init__(self) -> None:
        # user_id -> {mem_id -> MemoryRecord}
        self._data: dict[str, dict[str, MemoryRecord]] = {}

    def upsert(self, rec: MemoryRecord) -> None:
        self._data.setdefault(rec.user_id, {})[rec.mem_id] = rec

    def search(self, user_id: str, vec: List[float], top_k: int = 3) -> List[Tuple[MemoryRecord, float]]:
        bucket = self._data.get(user_id, {})
        # 只召回“现行”记忆（未被覆盖）
        scored = [
            (rec, _cosine(vec, rec.embedding))
            for rec in bucket.values()
            if rec.superseded_by is None and rec.embedding
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def delete(self, user_id: str, mem_ids: List[str]) -> None:
        bucket = self._data.get(user_id, {})
        for mid in mem_ids:
            bucket.pop(mid, None)

    def list_by_user(self, user_id: str) -> List[MemoryRecord]:
        return list(self._data.get(user_id, {}).values())

    @property
    def backend(self) -> str:
        return "memory"


class QdrantMemoryStore:
    """Qdrant 记忆库。payload 过滤 user_id 实现多租户隔离（运维最省）。"""

    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.mem_collection
        self._ready = False

    def _ensure(self, dim: int) -> None:
        if self._ready:
            return
        from qdrant_client.models import Distance, VectorParams

        names = [c.name for c in self.client.get_collections().collections]
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._ready = True

    @staticmethod
    def _pid(mem_id: str) -> str:
        # 与 rag/qdrant_store 完全一致的 id 生成方式 → 同 mem_id 天然 upsert 覆盖
        return str(uuid.uuid5(uuid.NAMESPACE_URL, mem_id))

    def upsert(self, rec: MemoryRecord) -> None:
        from qdrant_client.models import PointStruct

        if not rec.embedding:
            return
        self._ensure(len(rec.embedding))
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=self._pid(rec.mem_id), vector=rec.embedding,
                                payload=rec.to_payload())],
        )

    def _user_filter(self, user_id: str):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

    def search(self, user_id: str, vec: List[float], top_k: int = 3) -> List[Tuple[MemoryRecord, float]]:
        try:
            hits = self.client.search(
                collection_name=self.collection, query_vector=vec,
                query_filter=self._user_filter(user_id), limit=top_k,
            )
        except Exception:
            return []  # collection 尚未创建（冷启动）等情况：视为无记忆
        out: List[Tuple[MemoryRecord, float]] = []
        for h in hits:
            p = h.payload or {}
            if p.get("superseded_by"):
                continue
            out.append((MemoryRecord.from_payload(p), float(h.score)))
        return out

    def delete(self, user_id: str, mem_ids: List[str]) -> None:
        from qdrant_client.models import PointIdsList

        if not mem_ids:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[self._pid(m) for m in mem_ids]),
        )

    def list_by_user(self, user_id: str) -> List[MemoryRecord]:
        try:
            res, _ = self.client.scroll(
                collection_name=self.collection, scroll_filter=self._user_filter(user_id),
                with_payload=True, with_vectors=True, limit=10000,
            )
        except Exception:
            return []
        return [MemoryRecord.from_payload(p.payload or {}, p.vector or []) for p in res]

    @property
    def backend(self) -> str:
        return "qdrant"


_memory_store = None


def get_memory_store():
    """单例工厂：与 rag.store_factory.get_store() 同构的回退逻辑。"""
    global _memory_store
    if _memory_store is not None:
        return _memory_store
    if getattr(settings, "vector_backend", "memory") == "qdrant":
        try:
            store = QdrantMemoryStore()
            store.client.get_collections()  # 探活
            print(f"[memory] using Qdrant @ {settings.qdrant_url} / {settings.mem_collection}")
            _memory_store = store
            return _memory_store
        except Exception as e:  # noqa: BLE001
            print(f"[memory] Qdrant unavailable ({e}); fallback to memory")
    _memory_store = InMemoryMemoryStore()
    return _memory_store
