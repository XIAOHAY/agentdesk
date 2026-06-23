"""记忆召回评测：memory hit@k —— “该被想起的记忆有没有被召回到 Top-k”。

用法：python -m eval.run_memory_eval [k]
对每条样本：先植入 seed 记忆 → 用 query 召回 Top-k → 判断 expect_mem 是否命中。
复用 LongTermMemory.write/retrieve，与现有 eval/ 体系同风格，把记忆层接进评测。
"""
from __future__ import annotations

import json
import os
import sys
from typing import List

from app.config import settings
from app.memory.long_term import LongTermMemory
from app.memory.schema import KIND_PREFERENCE

DATASET = "eval/memory_dataset.jsonl"


def load_dataset() -> List[dict]:
    rows = []
    with open(DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _hit(recalled_texts: List[str], expect: str) -> bool:
    return any(expect in t for t in recalled_texts)


def evaluate(rows: List[dict], ks: List[int]) -> dict:
    lm = LongTermMemory()
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    for row in rows:
        uid = row["user_id"]
        # 植入种子记忆
        lm.write(uid, [{"kind": KIND_PREFERENCE, "text": s} for s in row["seed"]])
        # 召回
        recs = lm.retrieve(uid, row["query"], top_k=max_k)
        texts = [r.text for r in recs]
        for k in ks:
            if _hit(texts[:k], row["expect_mem"]):
                hits[k] += 1
    n = len(rows) or 1
    return {f"hit@{k}": round(hits[k] / n, 3) for k in ks}


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ks = sorted({1, 3, k})
    rows = load_dataset()
    print(f"记忆评测集 {len(rows)} 条 | use_llm={settings.use_llm} | "
          f"backend={settings.vector_backend}")
    results = evaluate(rows, ks)
    print(f"{'metric':<10}{'value':>8}")
    print("-" * 18)
    for key, val in results.items():
        print(f"{key:<10}{val:>8.3f}")

    os.makedirs("eval/reports", exist_ok=True)
    with open("eval/reports/memory_latest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n已写入 eval/reports/memory_latest.json")


if __name__ == "__main__":
    main()
