"""生成侧评测：跑样本问题，统计平均 faithfulness、通过率与评判方法。
用法：python -m eval.run_faithfulness [样本数，默认8]"""
from __future__ import annotations

import json
import sys

from app.graph.build_graph import run_query


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = [json.loads(l) for l in open("eval/dataset.jsonl", encoding="utf-8") if l.strip()][:n]
    scores, passed, methods = [], 0, set()
    for r in rows:
        s = run_query(r["question"])
        v = s.get("verify", {})
        scores.append(v.get("score", 0.0))
        passed += 1 if v.get("faithful") else 0
        methods.add(v.get("method", "?"))
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"样本 {len(rows)} | 评判方法 {methods}")
    print(f"平均 faithfulness = {avg:.3f}")
    print(f"通过率(faithful)  = {passed}/{len(rows)} = {passed/len(rows):.2%}")


if __name__ == "__main__":
    main()
