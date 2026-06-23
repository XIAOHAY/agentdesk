"""组装 LangGraph：memory_retrieve → planner → retrieval → tool → writer → critic
→(retry?)→ memory_write →(summarize?)→ END。
langgraph 不可用时退化为等价的顺序+循环执行，保证任何环境可演示。"""
from __future__ import annotations

import uuid

from app.config import settings
from app.graph.state import AgentState
from app.graph.nodes import (
    memory_retrieve_node, planner_node, retrieval_node, tool_node,
    writer_node, critic_node, should_retry,
    memory_write_node, summarize_node, need_summarize_edge,
)


def _build_compiled():
    from langgraph.graph import StateGraph, END

    g = StateGraph(AgentState)
    g.add_node("memory_retrieve", memory_retrieve_node)
    g.add_node("planner", planner_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("tool", tool_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)
    g.add_node("memory_write", memory_write_node)
    g.add_node("summarize", summarize_node)
    g.set_entry_point("memory_retrieve")
    g.add_edge("memory_retrieve", "planner")
    g.add_edge("planner", "retrieval")
    g.add_edge("retrieval", "tool")
    g.add_edge("tool", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", should_retry,
                            {"retry": "retrieval", "end": "memory_write"})
    g.add_conditional_edges("memory_write", need_summarize_edge,
                            {"summarize": "summarize", "end": END})
    g.add_edge("summarize", END)
    return g.compile()


_compiled = None


def _init_state(query: str, user_id: str, session_id: str) -> AgentState:
    return {"query": query, "user_id": user_id, "session_id": session_id,
            "trace": [], "iterations": 0}


def _run_sequential(state: AgentState) -> AgentState:
    state.update(memory_retrieve_node(state))
    state.update(planner_node(state))
    while True:
        state.update(retrieval_node(state))
        state.update(tool_node(state))
        state.update(writer_node(state))
        state.update(critic_node(state))
        if should_retry(state) == "end":
            break
    state.update(memory_write_node(state))
    if need_summarize_edge(state) == "summarize":
        state.update(summarize_node(state))
    return state


def run_query(query: str, user_id: str = "anonymous",
              session_id: str | None = None) -> AgentState:
    session_id = session_id or uuid.uuid4().hex
    global _compiled
    try:
        if _compiled is None:
            _compiled = _build_compiled()
        result = _compiled.invoke(_init_state(query, user_id, session_id))
    except Exception:
        result = _run_sequential(_init_state(query, user_id, session_id))
    try:
        from app.graph.trace_log import log_trace
        log_trace(result)  # 落盘 eval/reports/traces.jsonl；失败不影响返回
    except Exception:
        pass
    return result
