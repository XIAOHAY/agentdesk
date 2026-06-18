"""生成「难」语料 + 配套评测集：一批主题高度相似、仅靠型号/数字/地域区分的产品文档。

设计目的：让纯向量检索容易在相似文档间混淆，而 BM25 能靠精确的型号/数字命中，
从而让 hybrid（向量+BM25, RRF）+ Rerank 相对朴素向量跑出可见的 hit@k/MRR 提升。

运行：python -m scripts.gen_corpus   （会写入 data/docs/ 与 eval/dataset.jsonl）
"""
from __future__ import annotations

import json
import os
import random

random.seed(7)

DOCS_DIR = "data/docs"
DATASET = "eval/dataset.jsonl"

REGIONS = ["华东", "华北", "华南", "西南", "华中"]
PLAN_BASE = "本套餐为企业级云存储与算力解决方案，提供高可用对象存储、弹性计算与统一控制台，适用于中大型团队的数据密集型业务。"


def gen():
    os.makedirs(DOCS_DIR, exist_ok=True)
    rows = []
    files = []
    for i in range(20):
        code = f"AC-{100 + i}"
        region = REGIONS[i % len(REGIONS)]
        price = 1980 + i * 130            # 月费
        sla = round(99.50 + (i % 5) * 0.09, 2)   # SLA %
        storage = 200 + i * 50            # 存储上限 GB
        qps = 1000 + i * 250              # 并发上限
        fname = f"plan_{code}.md"
        content = (
            f"# 云服务套餐 {code}\n\n"
            f"## 概述\n{PLAN_BASE}\n\n"
            f"## 关键参数\n"
            f"- 套餐型号：{code}\n"
            f"- 部署地域：{region}\n"
            f"- 月费：{price} 元\n"
            f"- SLA 可用性：{sla}%\n"
            f"- 存储上限：{storage} GB\n"
            f"- 并发上限：{qps} QPS\n\n"
            f"## 说明\n型号 {code} 适用于 {region} 区域客户，"
            f"如需跨区域容灾请联系商务升级。\n"
        )
        with open(os.path.join(DOCS_DIR, fname), "w", encoding="utf-8") as f:
            f.write(content)
        files.append(fname)

        # 三类问题：精确型号(BM25友好) / 数字(混合) / 语义改写(向量友好)
        rows.append({"question": f"{code} 套餐的 SLA 可用性是多少？", "relevant_docs": [fname]})
        if i % 2 == 0:
            rows.append({"question": f"型号 {code} 的并发上限和存储上限分别是多少？", "relevant_docs": [fname]})
        if i % 3 == 0:
            rows.append({"question": f"部署在{region}、月费 {price} 元的那个套餐叫什么型号？", "relevant_docs": [fname]})

    random.shuffle(rows)
    with open(DATASET, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"生成 {len(files)} 篇产品文档 -> {DOCS_DIR}")
    print(f"生成 {len(rows)} 条评测问题 -> {DATASET}")


if __name__ == "__main__":
    gen()
