"""LLM 与 Embedding 的统一封装。

有 OPENAI_API_KEY 时调真实 API；否则用确定性 fallback（哈希向量 + 模板拼接），
保证无网络/无 key 也能端到端跑通。Embedding 带缓存（Redis/内存）。
"""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from app.config import settings

EMBED_DIM = 256


def _tokens(text: str) -> List[str]:
    text = text.lower()
    toks = text.split()
    chars = [c for c in text if not c.isspace()]
    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return toks + bigrams


def _hash_embedding(text: str, dim: int = EMBED_DIM) -> List[float]:
    vec = np.zeros(dim, dtype=np.float32)
    for token in _tokens(text):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _embed_raw(texts: List[str]) -> List[List[float]]:
    if not settings.use_llm:
        return [_hash_embedding(t) for t in texts]
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                    timeout=30.0, max_retries=1)  # 防接口挂起导致前端无限转圈
    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """带缓存的批量 embedding：命中缓存的跳过，未命中的批量计算后回填。"""
    from app.rag.cache import cache

    out: List[List[float]] = [None] * len(texts)  # type: ignore
    miss_idx, miss_texts = [], []
    for i, t in enumerate(texts):
        c = cache.get(t)
        if c is not None:
            out[i] = c
        else:
            miss_idx.append(i)
            miss_texts.append(t)
    if miss_texts:
        computed = _embed_raw(miss_texts)
        for i, t, vec in zip(miss_idx, miss_texts, computed):
            out[i] = vec
            cache.set(t, vec)
    return out


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]


def chat(system: str, user: str) -> str:
    if not settings.use_llm:
        return "[offline] no OPENAI_API_KEY; stitched answer from evidence:\n\n" + user
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                    timeout=30.0, max_retries=1)  # 防接口挂起导致前端无限转圈
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""
