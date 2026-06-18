"""演示真实 MCP stdio 传输：握手 -> tools/list -> tools/call。
运行：python -m scripts.mcp_demo"""
from __future__ import annotations

from app.tools.mcp_client import MCPClient


def main() -> None:
    with MCPClient() as c:
        info = c.initialize()  # 幂等
        print("server:", info["serverInfo"], "proto:", info["protocolVersion"])
        print("tools:", [t["name"] for t in c.list_tools()])
        print("calc:", c.call_tool("calculator", {"expression": "(210-205)/205*100"}))
        print("bad :", c.call_tool("nope", {}))           # 错误经协议回传
        print("inj :", c.call_tool("calculator", {"expression": "__import__('os')"}))


if __name__ == "__main__":
    main()
