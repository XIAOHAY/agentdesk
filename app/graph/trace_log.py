"""执行链落盘：每次 run_query 结束，把本轮 trace 追加写到 JSONL（一行一条，便于 grep/回放）。

设计：
- 默认开（settings.trace_log / 环境变量 TRACE_LOG；设 0 关闭）。
- 只写 JSON 可序列化字段；evidence 等含 dataclass 的不写（trace 里已含命中摘要）。
- 任何异常都吞掉，绝不影响主问答流程。
"""
from __future__ import annotations

import json
import os
import time

from app.config import settings

TRACE_PATH = "eval/reports/traces.jsonl"


def log_trace(state: dict) -> None:
    if not getattr(settings, "trace_log", True):
        return
    try:
        os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "user_id": state.get("user_id"),
            "session_id": state.get("session_id"),
            "query": state.get("query"),
            "answer": state.get("answer"),
            "iterations": state.get("iterations"),
            "verify": state.get("verify"),
            "citations": state.get("citations"),
            "recalled_memories": state.get("recalled_memories"),
            "memory_writes": state.get("memory_writes"),
            "trace": state.get("trace"),
        }
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 落盘失败不影响主流程
