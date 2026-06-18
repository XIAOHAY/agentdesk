"""检索评测：对比不同检索配置的 hit@k / MRR。

用法：python -m eval.run_eval
在 doc 级别评测（相关性标注为 relevant_docs），对 chunk 切分鲁棒。

输出三种配置对比，量化「混合检索 + Rerank」相对朴素向量检索的提升，
这就是简历里 “X% -> Y%” 的数据来源。
"""
from __future__ import annotations

import json
import os
from typing import List

from app.config import settings
from app.rag.retriever import Retriever

DATASET = "eval/dataset.jsonl"
TOP_K = 5

CONFIGS = [
    ("vector",        dict(mode="vector", use_rerank=False)),
    ("hybrid",        dict(mode="hybrid", use_rerank=False)),
    ("hybrid+rerank", dict(mode="hybrid", use_rerank=True)),
]


def load_dataset() -> List[dict]:
    rows = []
    with open(DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(retriever: Retriever, rows: List[dict], cfg: dict) -> dict:
    hit, rr = 0, 0.0
    for row in rows:
        rel = set(row["relevant_docs"])
        ev = retriever.retrieve(row["question"], top_k=TOP_K, **cfg)
        docs = [e.doc_id for e in ev]
        if rel & set(docs):
            hit += 1
        # MRR：首个相关文档的排名
        for i, d in enumerate(docs):
            if d in rel:
                rr += 1.0 / (i + 1)
                break
    n = len(rows)
    return {"hit@k": hit / n, "MRR": rr / n}


def main() -> None:
    rows = load_dataset()
    retriever = Retriever()
    print(f"评测集 {len(rows)} 条 | top_k={TOP_K} | use_llm={settings.use_llm}")
    print(f"{'config':<16}{'hit@k':>10}{'MRR':>10}")
    print("-" * 36)
    results = {}
    for name, cfg in CONFIGS:
        m = evaluate(retriever, rows, cfg)
        results[name] = m
        print(f"{name:<16}{m['hit@k']:>10.3f}{m['MRR']:>10.3f}")

    os.makedirs("eval/reports", exist_ok=True)
    with open("eval/reports/latest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n已写入 eval/reports/latest.json")


if __name__ == "__main__":
    main()
