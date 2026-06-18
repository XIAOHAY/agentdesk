"""Qdrant 向量库适配器。与内存版 VectorStore 接口一致，可直接替换。

接口：add / search / save / load / chunks / __len__
依赖 qdrant-client（可选）。集合不存在时按向量维度自动创建。
"""
from __future__ import annotations

import uuid
from typing import List

from app.config import settings
from app.rag.store import Chunk


class QdrantStore:
    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection

    def _ensure(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, chunks: List[Chunk]) -> None:
        from qdrant_client.models import PointStruct
        if not chunks:
            return
        self._ensure(len(chunks[0].embedding))
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                vector=c.embedding,
                payload={"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text},
            )
            for c in chunks
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vec: List[float], top_k: int = 5):
        hits = self.client.search(
            collection_name=self.collection, query_vector=query_vec, limit=top_k
        )
        out = []
        for h in hits:
            p = h.payload or {}
            out.append((Chunk(p["doc_id"], p["chunk_id"], p["text"], []), float(h.score)))
        return out

    @property
    def chunks(self) -> List[Chunk]:
        res, _ = self.client.scroll(
            collection_name=self.collection, with_payload=True,
            with_vectors=True, limit=10000,
        )
        return [Chunk(p.payload["doc_id"], p.payload["chunk_id"],
                      p.payload["text"], p.vector or []) for p in res]

    def save(self, path: str) -> None:
        pass  # Qdrant 已持久化

    def load(self, path: str) -> None:
        pass  # 数据已在 Qdrant 中

    def __len__(self) -> int:
        try:
            return self.client.count(collection_name=self.collection).count
        except Exception:
            return 0
