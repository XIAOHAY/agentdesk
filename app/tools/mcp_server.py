"""MCP 风格服务端：JSON-RPC 2.0 over stdio（纯标准库，无第三方依赖）。

这是 MCP stdio 传输的底层规范：服务端从 stdin 逐行读取 JSON-RPC 请求，
向 stdout 逐行写出响应。实现三个方法：
  - initialize        : 握手，返回协议版本与能力
  - tools/list        : 工具发现
  - tools/call        : 调用工具 {name, arguments} -> {content:[...], isError}

生产可直接换成官方 `mcp` SDK（from mcp.server import Server），
方法契约一致；这里手写是为了无依赖、可离线演示传输层原理。

运行：python -m app.tools.mcp_server   （通过管道与客户端通信）
"""
from __future__ import annotations

import json
import sys

from app.tools.builtins import registry

PROTOCOL_VERSION = "2024-11-05"


def _handle(req: dict) -> dict | None:
    rid = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    # 通知（无 id）不需要响应
    if method == "notifications/initialized":
        return None

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": "agentdesk-tools", "version": "0.1.0"},
            "capabilities": {"tools": {}},
        }
    elif method == "tools/list":
        tools = []
        for t in registry.list_tools():
            props = {k: {"type": "string" if v == "str" else "number"}
                     for k, v in t["params"].items()}
            tools.append({
                "name": t["name"],
                "description": t["description"],
                "inputSchema": {"type": "object", "properties": props,
                                "required": t["required"]},
            })
        result = {"tools": tools}
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        out = registry.call_tool(name, args)
        text = out.get("result") if out.get("ok") else f"ERROR: {out.get('error')}"
        return {"jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": str(text)}],
                           "isError": not out.get("ok")}}
    else:
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"method not found: {method}"}}

    return {"jsonrpc": "2.0", "id": rid, "result": result}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
