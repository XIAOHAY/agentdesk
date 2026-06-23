"""记忆演化：写入去重 / 冲突更新 / 过期淘汰。

作用时机：每次写入长期记忆时调用 resolve_write 决定 INSERT / UPDATE / CONFLICT_OVERWRITE；
写入后调用 evict_if_needed 做 TTL + 容量 LRU 淘汰。

阈值与上限全部走 settings，可在 .env 调；冲突覆盖只对 fact 生效且保留审计痕迹（不物理删旧值），
对应 JD「Memory 的持续进化链路」。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.config import settings
from app.memory.schema import KIND_EVENT, KIND_FACT, MemoryRecord, now

# 决策枚举
INSERT = "insert"
UPDATE = "update"               # 去重：同一条，刷新 use_count/updated_at
CONFLICT_OVERWRITE = "conflict_overwrite"  # 同一事实新取值：覆盖，旧值留痕


@dataclass
class WriteDecision:
    action: str
    target: Optional[MemoryRecord] = None  # UPDATE/CONFLICT 时为被命中的旧记录


class MemoryEvolution:
    def __init__(self, store) -> None:
        self.store = store
        self.dedup = float(getattr(settings, "mem_dedup_threshold", 0.92))
        self.conflict = float(getattr(settings, "mem_conflict_threshold", 0.80))
        self.event_ttl = float(getattr(settings, "mem_event_ttl_days", 30)) * 86400.0
        self.max_per_user = int(getattr(settings, "mem_max_per_user", 500))

    def resolve_write(self, new: MemoryRecord) -> WriteDecision:
        """根据与已有记忆的最相似项决定写入动作。"""
        hits = self.store.search(new.user_id, new.embedding, top_k=1)
        if not hits:
            return WriteDecision(INSERT)
        rec, sim = hits[0]
        if sim >= self.dedup:
            return WriteDecision(UPDATE, rec)
        if sim >= self.conflict and rec.kind == new.kind == KIND_FACT:
            return WriteDecision(CONFLICT_OVERWRITE, rec)
        return WriteDecision(INSERT)

    def apply(self, new: MemoryRecord) -> MemoryRecord:
        """执行写入决策，返回最终落库的现行记录。"""
        decision = self.resolve_write(new)

        if decision.action == UPDATE and decision.target is not None:
            old = decision.target
            old.use_count += 1
            old.updated_at = now()
            old.last_used_at = now()
            self.store.upsert(old)
            return old

        if decision.action == CONFLICT_OVERWRITE and decision.target is not None:
            old = decision.target
            # 旧值留审计痕迹：标记被新值取代（不物理删）
            old.superseded_by = new.mem_id
            old.updated_at = now()
            self.store.upsert(old)
            new.version = old.version + 1
            self.store.upsert(new)
            return new

        # INSERT
        self.store.upsert(new)
        return new

    def evict_if_needed(self, user_id: str) -> List[str]:
        """TTL 过期 + 容量 LRU 淘汰，返回被删除的 mem_id。"""
        recs = [r for r in self.store.list_by_user(user_id) if r.superseded_by is None]
        to_delete: List[str] = []
        t = now()

        # 1) TTL：仅 event 类
        survivors: List[MemoryRecord] = []
        for r in recs:
            if r.kind == KIND_EVENT and (t - r.created_at) > self.event_ttl:
                to_delete.append(r.mem_id)
            else:
                survivors.append(r)

        # 2) 容量上限：LRU（last_used_at 升序）+ use_count 加权保护热点
        if len(survivors) > self.max_per_user:
            survivors.sort(key=lambda r: (r.last_used_at + r.use_count * 3600.0))
            overflow = len(survivors) - self.max_per_user
            to_delete.extend(r.mem_id for r in survivors[:overflow])

        if to_delete:
            self.store.delete(user_id, to_delete)
        return to_delete
