"""记忆层（Memory Layer）。

三层结构：
- short_term：短期工作记忆（对话 buffer + 滚动 summary），Redis/内存回退。
- long_term ：长期记忆（用户偏好/事实 抽取→向量化→检索注入），Qdrant/内存回退。
- evolution ：记忆演化（写入去重 / 冲突更新 / 过期淘汰）。

全部遵循项目纪律：无 key / 无 Qdrant / 无 Redis 时自动回退，端到端可运行。
"""
from __future__ import annotations

from app.memory.schema import MemoryRecord, WorkingMemory
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory

__all__ = ["MemoryRecord", "WorkingMemory", "LongTermMemory", "ShortTermMemory"]
