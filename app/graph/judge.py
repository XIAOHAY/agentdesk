"""Faithfulness 评判：有 LLM 时用 LLM-as-judge 做事实核查打分，否则回退启发式。

返回 {faithful, score, method, reason?, detail}。score∈[0,1]，表示答案被证据支撑的程度。
detail 给前端展示"这次是怎么算的"（命中/答案词数、是否用到工具、证据条数）。
LLM-judge 更接近 RAGAS 的 faithfulness 思路（拆 claim → 逐条验证），这里用一次性打分简化。
"""
from __future__ import annotations

import json
import re

from app.config import settings
from app.llm import chat
from app.rag.tokenize import tokenize

THRESHOLD = 0.6


def _heuristic(answer: str, evidence, tool_text: str = "") -> float:
    ans = set(tokenize(answer))
    if not ans:
        return 0.0
    ev = set()
    for e in evidence:
        ev |= set(tokenize(e.text))
    if tool_text:  # 工具输出（如 kb_stats 的数字）同样算作"支撑"
        ev |= set(tokenize(tool_text))
    return len(ans & ev) / len(ans)


def judge(question: str, answer: str, evidence, tool_results=None) -> dict:
    # tool_results 是 kb_stats / calculator 等工具的输出，同样是"事实依据"，必须纳入忠实度判定，
    # 否则工具类正确答案会因"检索证据里没有该数字"被误判为不可信、触发无谓重试。
    # 该参数缺省（None）时行为与改动前完全一致，向后兼容。
    tool_text = "\n".join(
        r["out"]["result"] for r in (tool_results or [])
        if isinstance(r.get("out"), dict) and r["out"].get("ok") and r["out"].get("result")
    )
    n_evidence = len(list(evidence))

    if settings.use_llm and answer.strip():
        ctx = "\n\n".join(f"[{e.chunk_id}] {e.text}" for e in evidence)
        if tool_text:
            ctx += f"\n\n[工具结果]\n{tool_text}"
        system = (
            "你是严格的事实核查员。判断【答案】中的每条事实是否都能由【证据】支撑。"
            "【证据】包含检索资料与工具结果，工具结果（如统计数字）视为权威依据。"
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
                    "reason": str(obj.get("reason", ""))[:200], "method": "llm",
                    "detail": {"used_tool": bool(tool_text), "n_evidence": n_evidence}}
        except Exception:
            pass  # 解析失败则回退启发式，保证健壮

    # —— 启发式兜底：顺带产出推导明细供前端展示 ——
    ans_set = set(tokenize(answer))
    ev_set = set()
    for e in evidence:
        ev_set |= set(tokenize(e.text))
    if tool_text:
        ev_set |= set(tokenize(tool_text))
    score = (len(ans_set & ev_set) / len(ans_set)) if ans_set else 0.0
    return {"faithful": score >= THRESHOLD, "score": round(score, 3), "method": "heuristic",
            "detail": {"n_answer": len(ans_set), "n_match": len(ans_set & ev_set),
                       "used_tool": bool(tool_text), "n_evidence": n_evidence}}
