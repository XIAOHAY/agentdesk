"""LangGraph 全局 State 定义。节点共享、累积写入。"""
from __future__ import annotations

from typing import List, TypedDict

from app.rag.retriever import Evidence


class AgentState(TypedDict, total=False):
    query: str                 # 用户原始问题
    plan: str                  # 可读的检索意图拼接
    queries: List[str]         # 改写后的多条检索 query
    evidence: List[Evidence]   # 检索证据
    tool_results: List[dict]   # 工具调用结果
    answer: str                # 最终答案
    citations: List[str]       # 引用的 chunk_id
    verify: dict               # critic 的 faithfulness 判定
    iterations: int            # 已重试轮数（防死循环）
    trace: List[dict]          # 执行链记录
