"""FastAPI 入口：/ (Web UI) · /chat · /health。"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import List

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.graph.build_graph import run_query

app = FastAPI(title="AgentDesk", version="0.4.0")

_WEB = os.path.join(os.path.dirname(__file__), "..", "web", "index.html")


class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    evidence: list
    tool_results: list
    verify: dict
    iterations: int
    trace: list
    recalled_memories: list
    memory_writes: list


@app.get("/")
def home():
    return FileResponse(_WEB)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    s = run_query(req.query, user_id=req.user_id, session_id=req.session_id)
    ev = [asdict(e) if hasattr(e, "__dataclass_fields__") else e for e in s.get("evidence", [])]
    ev_view = [{"chunk_id": e["chunk_id"], "doc_id": e["doc_id"],
                "score": round(e["score"], 4), "text": e["text"][:200]} for e in ev]
    return ChatResponse(
        answer=s.get("answer", ""),
        citations=s.get("citations", []),
        evidence=ev_view,
        tool_results=s.get("tool_results", []),
        verify=s.get("verify", {}),
        iterations=s.get("iterations", 0),
        trace=s.get("trace", []),
        recalled_memories=s.get("recalled_memories", []),
        memory_writes=s.get("memory_writes", []),
    )
