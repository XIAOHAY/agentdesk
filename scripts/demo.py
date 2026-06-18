"""端到端 Demo：跑 3 个代表性场景，打印执行链/工具/反思结果。供录屏与自测。
运行：python -m scripts.demo"""
from __future__ import annotations

from app.graph.build_graph import run_query

SCENARIOS = [
    ("知识问答(命中年假政策)", "入职满三年有多少天年假？"),
    ("工具调用(计算器)", "(210-205)/205*100"),
    ("工具调用(知识库统计)", "知识库里有多少个文档？"),
]


def _fmt_nodes(trace):
    out = []
    for t in trace:
        n = t["node"]
        if n == "retrieval":
            n += f"#{t.get('iter')}"
        if n == "tool" and t.get("called"):
            n += f"({','.join(t['called'])})"
        out.append(n)
    return " -> ".join(out)


def main() -> None:
    for title, q in SCENARIOS:
        print("=" * 60)
        print(f"场景：{title}\n问题：{q}")
        s = run_query(q)
        print("执行链：", _fmt_nodes(s["trace"]))
        if s.get("tool_results"):
            for r in s["tool_results"]:
                print("工具：", r["tool"], "->", r["out"])
        print("证据TOP：", [(e.doc_id, round(e.score, 2)) for e in s["evidence"][:2]])
        print("反思：", s.get("verify"), "| 迭代：", s.get("iterations"))
        print("引用：", s.get("citations"))
    print("=" * 60)


if __name__ == "__main__":
    main()
