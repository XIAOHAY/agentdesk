"""按配置选择向量库后端；Qdrant 不可用时回退内存库，保证任何环境可运行。"""
from __future__ import annotations

from app.config import settings
from app.rag.store import VectorStore


def get_store():
    if getattr(settings, "vector_backend", "memory") == "qdrant":
        try:
            from app.rag.qdrant_store import QdrantStore
            store = QdrantStore()
            print(f"[store] using Qdrant @ {settings.qdrant_url}")
            return store
        except Exception as e:
            print(f"[store] Qdrant unavailable ({e}); fallback to memory")
    return VectorStore()
