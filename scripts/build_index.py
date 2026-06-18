"""建索引脚本：python -m scripts.build_index"""
from __future__ import annotations

from app.rag.indexer import build_index, INDEX_PATH


def main() -> None:
    store = build_index()
    print(f"已建索引：{len(store)} 个 chunk -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
