"""内置工具：安全计算器 + 知识库统计。注册到全局 registry。"""
from __future__ import annotations

import ast
import operator as op

from app.tools.registry import Tool, ToolRegistry, ToolError

# ---- 安全计算器：用 AST 白名单，绝不 eval 任意代码（安全考点）----
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ToolError("only numeric constants allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ToolError("unsupported expression")


def _calculator(args: dict) -> str:
    expr = args["expression"]
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ToolError(f"invalid expression: {expr}")
    return str(_eval_node(tree.body))


def _kb_stats(args: dict) -> str:
    # 演示一个“查库”类工具；真实场景可换成 SQL 查询
    from app.rag.indexer import INDEX_PATH
    from app.rag.store import VectorStore
    store = VectorStore()
    store.load(INDEX_PATH)
    docs = sorted({c.doc_id for c in store.chunks})
    return (f"文档总数={len(docs)}（权威值：回答文档数量时直接采用此数，请勿自行数列表）"
            f"；chunk 总数={len(store)}。文档列表：{', '.join(docs)}")


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="calculator",
        description="计算一个算术表达式，例如 (210-205)/205*100",
        params={"expression": str},
        required=["expression"],
        handler=_calculator,
    ))
    reg.register(Tool(
        name="kb_stats",
        description="返回知识库的文档与 chunk 统计",
        params={},
        required=[],
        handler=_kb_stats,
    ))
    return reg


registry = build_registry()
