"""MCP 风格的本地工具层（雏形）。

为什么不直接用裸函数：面试关注的可靠性都集中在这层——
  1) 工具名校验（防工具幻觉：模型调用不存在的工具直接拒绝）
  2) 参数 schema 校验（缺参/类型错 → 结构化报错，不崩溃）
  3) 输出截断（防 tool 输出爆 token）
  4) 统一错误处理（失败转为可读 error，交回模型纠正）

后续把 call_tool 接到真正的 MCP Server（Stdio/HTTP 传输）即可，
list_tools()/call_tool() 的契约保持不变。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

MAX_TOOL_OUTPUT = 800  # 字符级输出上限，超出截断


class ToolError(Exception):
    pass


@dataclass
class Tool:
    name: str
    description: str
    # 参数名 -> 类型（极简 schema；生产可换 pydantic/jsonschema）
    params: Dict[str, type]
    handler: Callable[[dict], str]
    required: List[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> List[dict]:
        return [
            {"name": t.name, "description": t.description,
             "params": {k: v.__name__ for k, v in t.params.items()},
             "required": t.required}
            for t in self._tools.values()
        ]

    def call_tool(self, name: str, args: dict) -> dict:
        # 1) 工具名校验
        if name not in self._tools:
            return {"ok": False, "error": f"unknown tool: {name}",
                    "available": list(self._tools.keys())}
        tool = self._tools[name]
        args = args or {}

        # 2) 必填 & 类型校验
        for r in tool.required:
            if r not in args:
                return {"ok": False, "error": f"missing required param: {r}"}
        for k, v in args.items():
            if k in tool.params and not isinstance(v, tool.params[k]):
                return {"ok": False,
                        "error": f"param '{k}' expected {tool.params[k].__name__}, got {type(v).__name__}"}

        # 3) 执行 + 统一错误处理
        try:
            out = tool.handler(args)
        except ToolError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:  # 兜底，绝不让工具异常打断整条链
            return {"ok": False, "error": f"tool crashed: {e}"}

        # 4) 输出截断
        truncated = False
        if isinstance(out, str) and len(out) > MAX_TOOL_OUTPUT:
            out = out[:MAX_TOOL_OUTPUT] + " ...[truncated]"
            truncated = True
        return {"ok": True, "result": out, "truncated": truncated}
