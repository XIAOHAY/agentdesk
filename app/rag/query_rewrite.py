"""查询改写（multi-query）。

把用户问题扩展成多条检索 query，提升召回。
有 LLM 时让模型生成同义/拆解变体；离线时回退为原问题。
"""
from __future__ import annotations

import json
from typing import List

from app.config import settings
from app.llm import chat


def rewrite(query: str, n: int = 3) -> List[str]:
    if not settings.use_llm:
        return [query]
    system = (
        "你是检索查询改写器。把用户问题改写成 {n} 条语义等价或"
        "聚焦子意图的检索查询，便于向量/关键词检索。"
        "只输出 JSON 数组，例如 [\"q1\",\"q2\"]。".format(n=n)
    )
    try:
        raw = chat(system, query)
        start, end = raw.find("["), raw.rfind("]")
        variants = json.loads(raw[start : end + 1])
        variants = [v for v in variants if isinstance(v, str) and v.strip()]
        # 原问题始终保留在首位，去重
        out = [query] + [v for v in variants if v != query]
        return out[: n + 1]
    except Exception:
        return [query]
