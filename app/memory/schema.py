"""记忆层数据模型。

- MemoryRecord：一条长期记忆（含演化所需的审计字段）。
- WorkingMemory：一个会话的短期记忆（对话 buffer + 滚动 summary）。

mem_id 用 (user_id + 归一化文本) 的 md5，使「同一用户的同一句话」天然定位到同一条，
便于去重与 upsert 覆盖。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# 长期记忆类型：语义记忆（fact/preference 长期稳定）与情景记忆（event 可较快过期）
KIND_FACT = "fact"
KIND_PREFERENCE = "preference"
KIND_EVENT = "event"
VALID_KINDS = {KIND_FACT, KIND_PREFERENCE, KIND_EVENT}


def now() -> float:
    return time.time()


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def make_mem_id(user_id: str, text: str) -> str:
    raw = f"{user_id}:{normalize_text(text)}"
    return "mem:" + hashlib.md5(raw.encode("utf-8")).hexdigest()


@dataclass
class MemoryRecord:
    user_id: str
    text: str
    kind: str = KIND_FACT
    embedding: List[float] = field(default_factory=list)
    mem_id: str = ""
    created_at: float = field(default_factory=now)
    updated_at: float = field(default_factory=now)
    last_used_at: float = field(default_factory=now)
    use_count: int = 0
    version: int = 1
    superseded_by: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            self.kind = KIND_FACT
        if not self.mem_id:
            self.mem_id = make_mem_id(self.user_id, self.text)

    def to_payload(self) -> dict:
        """落库 payload（不含向量，向量单独存）。"""
        d = asdict(self)
        d.pop("embedding", None)
        return d

    @classmethod
    def from_payload(cls, payload: dict, embedding: Optional[List[float]] = None) -> "MemoryRecord":
        data = dict(payload)
        return cls(
            user_id=data.get("user_id", ""),
            text=data.get("text", ""),
            kind=data.get("kind", KIND_FACT),
            embedding=embedding or [],
            mem_id=data.get("mem_id", ""),
            created_at=float(data.get("created_at", now())),
            updated_at=float(data.get("updated_at", now())),
            last_used_at=float(data.get("last_used_at", now())),
            use_count=int(data.get("use_count", 0)),
            version=int(data.get("version", 1)),
            superseded_by=data.get("superseded_by"),
        )


@dataclass
class WorkingMemory:
    session_id: str
    messages: List[dict] = field(default_factory=list)   # [{"role","content","ts"}]
    running_summary: str = ""
    round_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkingMemory":
        return cls(
            session_id=d.get("session_id", ""),
            messages=list(d.get("messages", [])),
            running_summary=d.get("running_summary", ""),
            round_count=int(d.get("round_count", 0)),
        )
