"""LangGraph 全局 State 定义。节点共享、累积写入。"""
from __future__ import annotations

from typing import List, TypedDict

from app.rag.retriever import Evidence


class AgentState(TypedDict, total=False):
    query: str
    plan: str
    queries: List[str]
    evidence: List[Evidence]
    tool_results: List[dict]
    answer: str
    citations: List[str]
    verify: dict
    iterations: int
    trace: List[dict]
    # —— 记忆层 ——
    user_id: str
    session_id: str
    working_memory: dict
    recalled_memories: List[dict]
    memory_writes: List[dict]
