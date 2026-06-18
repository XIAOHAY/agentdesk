"""MCP 客户端：spawn 服务端子进程，走 stdio JSON-RPC 完成握手与调用。

用法：
    with MCPClient() as c:
        tools = c.list_tools()
        out = c.call_tool("calculator", {"expression": "1+2"})

生产可换成官方 mcp SDK 的 ClientSession（stdio_client / SSE）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from typing import List


class MCPClient:
    def __init__(self, server_module: str = "app.tools.mcp_server") -> None:
        self._proc = subprocess.Popen(
            [sys.executable, "-m", server_module],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._id = 0
        self._initialized = False

    def _rpc(self, method: str, params: dict | None = None, notify: bool = False):
        msg = {"jsonrpc": "2.0", "method": method}
        if not notify:
            self._id += 1
            msg["id"] = self._id
        if params is not None:
            msg["params"] = params
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        if notify:
            return None
        line = self._proc.stdout.readline()
        return json.loads(line)

    def initialize(self) -> dict:
        resp = self._rpc("initialize", {"protocolVersion": "2024-11-05",
                                        "clientInfo": {"name": "agentdesk", "version": "0.1.0"}})
        self._rpc("notifications/initialized", notify=True)
        self._initialized = True
        return resp["result"]

    def list_tools(self) -> List[dict]:
        if not self._initialized:
            self.initialize()
        return self._rpc("tools/list")["result"]["tools"]

    def call_tool(self, name: str, arguments: dict) -> dict:
        if not self._initialized:
            self.initialize()
        return self._rpc("tools/call", {"name": name, "arguments": arguments})["result"]

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    def __enter__(self) -> "MCPClient":
        self.initialize()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
