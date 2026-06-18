"""统一分词器（中英混合友好）。

中文无空格，用「字符 + 字符 bigram」；英文/数字用空格词。
BM25 与离线 fallback embedding 共用，保证一致性。
"""
from __future__ import annotations

import re
from typing import List

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    text = text.lower()
    words = _WORD.findall(text)
    cjk = [c for c in text if "一" <= c <= "鿿"]
    bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return words + cjk + bigrams
