"""长期记忆：生产（抽取→向量化→演化写入）与消费（检索召回→注入）。

抽取默认走「规则触发」（零成本、可解释），命中自述型模式（"我是…""我只看…""记住…"）
即抽取该子句为一条记忆。后续可平滑升级为 LLM 结构化抽取（接口不变）。
"""
from __future__ import annotations

import re
from typing import List

from app.llm import embed_query, embed_texts
from app.memory.evolution import MemoryEvolution
from app.memory.schema import (
    KIND_EVENT,
    KIND_FACT,
    KIND_PREFERENCE,
    MemoryRecord,
    now,
)
from app.memory.store import get_memory_store

# 子句切分符
_SPLIT = re.compile(r"[，。；、,.;\n]+")
# 抽取规则：模式 → 记忆类型。命中即把该子句作为一条记忆。
_RULES = [
    (re.compile(r"(我是|我叫|我的名字|我来自|我在.{0,8}(工作|上班)|我的职业)"), KIND_FACT),
    (re.compile(r"(只看|只关注|只要|偏好|喜欢|习惯|以后都|默认用|我倾向)"), KIND_PREFERENCE),
    (re.compile(r"(记住|请记住|备注|提醒我)"), KIND_FACT),
    (re.compile(r"(上次|刚才|今天|昨天).{0,12}(完成|提交|发生|做了)"), KIND_EVENT),
]
# 问句特征：纯提问不抽取（避免把问题当事实存进去）
_QUESTION = re.compile(r"(吗|呢|怎么|如何|为什么|多少|哪些|是不是|？|\?)\s*$")


class LongTermMemory:
    def __init__(self) -> None:
        self.store = get_memory_store()
        self.evolution = MemoryEvolution(self.store)

    # ---------- 生产 ----------
    def extract(self, query: str, answer: str = "") -> List[dict]:
        """从用户输入里抽取值得长期记住的偏好/事实（规则版）。"""
        items: List[dict] = []
        seen = set()
        for clause in _SPLIT.split(query or ""):
            c = clause.strip()
            if len(c) < 3 or c in seen:
                continue
            for pat, kind in _RULES:
                if pat.search(c) and not _QUESTION.search(c):
                    items.append({"kind": kind, "text": c})
                    seen.add(c)
                    break
        return items

    def write(self, user_id: str, items: List[dict]) -> List[MemoryRecord]:
        """把抽取项写入长期记忆，经 evolution 去重/冲突，最后做淘汰。"""
        if not user_id or not items:
            return []
        texts = [it["text"] for it in items]
        vectors = embed_texts(texts)
        written: List[MemoryRecord] = []
        for it, vec in zip(items, vectors):
            rec = MemoryRecord(user_id=user_id, text=it["text"],
                               kind=it.get("kind", KIND_FACT), embedding=vec)
            written.append(self.evolution.apply(rec))
        self.evolution.evict_if_needed(user_id)
        return written

    # ---------- 消费 ----------
    def retrieve(self, user_id: str, query: str, top_k: int = 3) -> List[MemoryRecord]:
        """召回与当前问题相关的长期记忆，并回写命中热度（last_used/use_count）。"""
        if not user_id or not query:
            return []
        qv = embed_query(query)
        hits = self.store.search(user_id, qv, top_k=top_k)
        recalled: List[MemoryRecord] = []
        for rec, _score in hits:
            rec.last_used_at = now()
            rec.use_count += 1
            self.store.upsert(rec)
            recalled.append(rec)
        return recalled
