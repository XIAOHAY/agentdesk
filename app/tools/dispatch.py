"""工具调用分发：根据 settings.use_mcp 选择 MCP(stdio) 或本地 registry。

默认走本地 registry（进程内、零开销）；设 USE_MCP=1 时改走真实 MCP 子进程，
两条路径返回统一结构 {ok, result/error}，graph 无感知。
"""
from __future__ import annotations

from app.config import settings
from app.tools.builtins import registry

_mcp_client = None


def _get_mcp():
    global _mcp_client
    if _mcp_client is None:
        from app.tools.mcp_client import MCPClient
        _mcp_client = MCPClient()
        _mcp_client.initialize()
    return _mcp_client


def call(name: str, args: dict) -> dict:
    if getattr(settings, "use_mcp", False):
        try:
            res = _get_mcp().call_tool(name, args)
            content = res.get("content", [])
            text = content[0]["text"] if content else ""
            if res.get("isError"):
                return {"ok": False, "error": text, "via": "mcp"}
            return {"ok": True, "result": text, "via": "mcp"}
        except Exception as e:
            return {"ok": False, "error": f"mcp transport failed: {e}", "via": "mcp"}
    out = registry.call_tool(name, args)
    out["via"] = "local"
    return out
