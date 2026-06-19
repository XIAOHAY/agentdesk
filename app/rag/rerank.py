"""Rerank（重排）：把召回回来的候选按「与 query 的相关性」重新排序，
决定最终喂给 writer 的证据顺序（top_k）。排序质量直接决定答案质量。

──────────────────────────────────────────────────────────────────────────
为什么之前用「词项重叠占比」（_overlap_score_legacy）
──────────────────────────────────────────────────────────────────────────
初版：score = |q∩d| / |q|（query 与 doc 的 token 交集占 query 的比例）。
动机：零依赖、可离线、接口先跑通，符合本项目「无 key 也能端到端跑」的定位。
实测它在本评测集已能把 hybrid 的 MRR 从 0.941 拉到 1.0——因为每条问题都含一个
逐字唯一的 token（型号 ID 或精确月费），gold 文档恰好多命中这一个词，overlap 占了优势。
局限：overlap 把「样板词」和「唯一区分词」等权看待——靠的是 gold 多匹配那一个词的
「数量优势」，而不是「这个词更有区分力」。一旦 query 里没有逐字唯一 token，或候选
长度参差导致匹配数被样板词拉平，它就只能按噪声排序。

──────────────────────────────────────────────────────────────────────────
踩过的坑：为什么不直接用「纯 IDF 加权」（已验证会退化，故放弃）
──────────────────────────────────────────────────────────────────────────
曾尝试把打分整体换成「命中词的 IDF 之和」，想让区分词主导。结果离线 eval 的
hybrid+rerank MRR 从 0.941 掉到 0.595。复盘原因：中文 query 分词后混入「是多少 /
可用 / 性是」等疑问词、助词 bigram，这些词在模板化的 plan 文档里几乎不出现 → IDF 很高；
一旦它们偶然出现在某篇 off-topic 文档（如 sample_company），该文档分数被抬到 gold 之上。
教训：IDF 会放大「稀有但无关」的噪声词，不能单独当主排序键。

──────────────────────────────────────────────────────────────────────────
为什么现在这么改：overlap 主键 + IDF 仅做同分裁决 + 可插拔 cross-encoder
──────────────────────────────────────────────────────────────────────────
1) 离线兜底 = 「匹配词数」为主 + 「IDF 之和」为辅（_lexical_score）。
   主键仍是命中的 query 词数（= overlap 排序，已实证 MRR 1.0，保证不退化）；
   IDF 归一化到 [0,1) 后只作小数位裁决：匹配词数相同的并列文档之间，
   命中更有区分力词（高 IDF）的排前面。因为 IDF 加成 < 1，绝不会跨越整数边界、
   不会把匹配更多的文档挤下去 → 既保留 overlap 的稳健，又补上它「同分排不出」的短板。
   仍然零依赖、可离线，是默认路径。
   用「候选池」而非「全库」统计 IDF：rerank 只需在召回回来的相似文档间做区分，
   候选池内的 DF 恰好放大彼此差异，且自包含、免改外部接口。
2) 可插拔 cross-encoder：设了环境变量 RERANK_MODEL（如 BAAI/bge-reranker-v2-m3）
   且依赖可用时，自动改用 cross-encoder 做 query-doc 语义相关性打分——这才是
   生产级正解（能处理改写/无逐字 token 的难例）；否则回退到 (1)。
   一份接口，离线能跑、上线能升级，默认不加载重模型，互不破坏。
"""
from __future__ import annotations

import math
import os
from collections import Counter
from typing import List

from app.rag.store import Chunk
from app.rag.tokenize import tokenize

# ── 可插拔 cross-encoder（默认不启用；设 RERANK_MODEL 才尝试加载）────────────
_CE_TRIED = False
_CE_MODEL = None  # type: ignore


def _get_cross_encoder():
    """惰性加载 cross-encoder；只尝试一次，失败则永久回退到离线打分。"""
    global _CE_TRIED, _CE_MODEL
    if _CE_TRIED:
        return _CE_MODEL
    _CE_TRIED = True
    model_name = os.environ.get("RERANK_MODEL", "").strip()
    if not model_name:
        return None  # 未配置 → 走离线兜底
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        _CE_MODEL = CrossEncoder(model_name)
    except Exception:
        _CE_MODEL = None  # 依赖缺失/加载失败 → 回退
    return _CE_MODEL


# ── 离线兜底：匹配词数为主 + IDF 同分裁决 ────────────────────────────────────
def _candidate_idf(candidates: List[Chunk]) -> dict:
    """在候选池内统计 IDF：样板词（每篇都有）→≈0，区分词（罕见）→高。"""
    n = len(candidates)
    df: Counter = Counter()
    for c in candidates:
        for term in set(tokenize(c.text)):
            df[term] += 1
    return {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}


def _lexical_score(query: str, text: str, idf: dict, idf_total: float) -> float:
    """score = 命中 query 词数  +  IDF 加成(∈[0,1))。
    整数部分 = overlap 排序（稳健、不退化）；小数部分仅在同分时按区分力裁决。"""
    q_terms = set(tokenize(query))
    if not q_terms:
        return 0.0
    matched = q_terms & set(tokenize(text))
    base = float(len(matched))
    bonus = (sum(idf.get(t, 0.0) for t in matched) / idf_total) if idf_total > 0 else 0.0
    return base + min(bonus, 0.999)  # 加成封顶 <1，绝不跨越整数边界


def _overlap_score_legacy(query: str, text: str) -> float:
    """【保留作对照】均等词项重叠占比；与 _lexical_score 的整数部分同序。"""
    q = set(tokenize(query))
    if not q:
        return 0.0
    return len(q & set(tokenize(text))) / len(q)


def rerank(query: str, candidates: List[Chunk], top_k: int = 5) -> List[tuple[Chunk, float]]:
    """对候选重排，返回 [(chunk, score), ...]（降序，截断 top_k）。
    优先 cross-encoder（若 RERANK_MODEL 配置且可用），否则离线兜底。"""
    if not candidates:
        return []

    ce = _get_cross_encoder()
    if ce is not None:
        try:
            scores = ce.predict([(query, c.text) for c in candidates])
            scored = [(c, float(s)) for c, s in zip(candidates, scores)]
            scored.sort(key=lambda x: -x[1])
            return scored[:top_k]
        except Exception:
            pass  # 运行期失败 → 落回离线兜底，保证不挂

    idf = _candidate_idf(candidates)
    idf_total = sum(idf.get(t, 0.0) for t in set(tokenize(query)))
    scored = [(c, _lexical_score(query, c.text, idf, idf_total)) for c in candidates]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]
