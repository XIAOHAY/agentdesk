"""短期工作记忆：对话 buffer + 滚动 summary。

策略：保留「running_summary + 最近 K 轮原文」；round_count 超过 N 触发摘要，
把旧轮次压成新 summary，只留最近 K 轮。存储优先 Redis（与 cache.py 同思路），
连不上回退进程内 dict。

ShortTermMemory 是无状态管理器，操作 WorkingMemory 数据对象。
"""
from __future__ import annotations

import json
from typing import Optional

from app.config import settings
from app.llm import chat
from app.memory.schema import WorkingMemory, now

_WM_TTL = 86400  # 会话级 TTL：1 天


class ShortTermMemory:
    def __init__(self) -> None:
        self._mem: dict[str, str] = {}
        self._redis = None
        url = getattr(settings, "redis_url", "")
        if url:
            try:
                import redis  # 可选依赖

                self._redis = redis.from_url(url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None
        self.window_k = int(getattr(settings, "mem_short_window_k", 4))
        self.summarize_every_n = int(getattr(settings, "mem_summarize_every_n", 8))

    @staticmethod
    def _key(session_id: str) -> str:
        return f"wm:{session_id}"

    def load(self, session_id: str) -> WorkingMemory:
        raw: Optional[str]
        if self._redis is not None:
            raw = self._redis.get(self._key(session_id))
        else:
            raw = self._mem.get(self._key(session_id))
        if raw:
            try:
                return WorkingMemory.from_dict(json.loads(raw))
            except Exception:
                pass
        return WorkingMemory(session_id=session_id)

    def persist(self, wm: WorkingMemory) -> None:
        raw = json.dumps(wm.to_dict(), ensure_ascii=False)
        if self._redis is not None:
            self._redis.set(self._key(wm.session_id), raw, ex=_WM_TTL)
        else:
            self._mem[self._key(wm.session_id)] = raw

    def append_turn(self, wm: WorkingMemory, user_text: str, assistant_text: str) -> None:
        wm.messages.append({"role": "user", "content": user_text, "ts": now()})
        wm.messages.append({"role": "assistant", "content": assistant_text, "ts": now()})
        wm.round_count += 1

    def need_summarize(self, wm: WorkingMemory) -> bool:
        return wm.round_count > 0 and wm.round_count % self.summarize_every_n == 0 \
            and len(wm.messages) > self.window_k * 2

    def build_context(self, wm: WorkingMemory) -> str:
        """拼出注入 prompt 的短期上下文：summary + 最近 K 轮原文。"""
        parts: list[str] = []
        if wm.running_summary:
            parts.append(f"【对话摘要】{wm.running_summary}")
        recent = wm.messages[-self.window_k * 2:]
        if recent:
            lines = [f"{m['role']}: {m['content']}" for m in recent]
            parts.append("【最近对话】\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def summarize(self, wm: WorkingMemory) -> WorkingMemory:
        """压缩旧轮次为滚动 summary，只保留最近 K 轮原文。"""
        keep = self.window_k * 2
        old = wm.messages[:-keep] if len(wm.messages) > keep else []
        if not old:
            return wm
        old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        system = (
            "把以下对话压缩成简洁要点，必须保留：已确认的事实、用户偏好、未决问题。"
            "不要编造，不要展开，输出 5 条以内。"
        )
        user = f"已有摘要：{wm.running_summary or '无'}\n\n待压缩对话：\n{old_text}"
        wm.running_summary = chat(system, user).strip()
        wm.messages = wm.messages[-keep:]
        return wm
