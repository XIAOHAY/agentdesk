"""Faithfulness 评判：有 LLM 时用 LLM-as-judge 做事实核查打分，否则回退启发式。

返回 {faithful, score, method, reason?}。score∈[0,1]，表示答案被证据支撑的程度。
LLM-judge 更接近 RAGAS 的 faithfulness 思路（拆 claim → 逐条验证），这里用一次性打分简化。
"""
from __future__ import annotations

import json
import re

from app.config import settings
from app.llm import chat
from app.rag.tokenize import tokenize

THRESHOLD = 0.6


def _heuristic(answer: str, evidence) -> float:
    ans = set(tokenize(answer))
    if not ans:
        return 0.0
    ev = set()
    for e in evidence:
        ev |= set(tokenize(e.text))
    return len(ans & ev) / len(ans)


def judge(question: str, answer: str, evidence) -> dict:
    if settings.use_llm and answer.strip():
        ctx = "\n\n".join(f"[{e.chunk_id}] {e.text}" for e in evidence)
        system = (
            "你是严格的事实核查员。判断【答案】中的每条事实是否都能由【证据】支撑。"
            "score = 被证据支撑的事实比例（0~1，1 表示完全支撑、无臆造）。"
            "只输出 JSON：{\"score\": 数字, \"reason\": \"简短说明\"}"
        )
        user = f"问题：{question}\n\n答案：{answer}\n\n证据：\n{ctx}"
        try:
            raw = chat(system, user)
            m = re.search(r"\{.*\}", raw, re.S)
            obj = json.loads(m.group(0))
            score = max(0.0, min(1.0, float(obj.get("score"))))
            return {"faithful": score >= THRESHOLD, "score": round(score, 3),
                    "reason": str(obj.get("reason", ""))[:200], "method": "llm"}
        except Exception:
            pass  # 解析失败则回退启发式，保证健壮
    score = _heuristic(answer, evidence)
    return {"faithful": score >= THRESHOLD, "score": round(score, 3), "method": "heuristic"}
