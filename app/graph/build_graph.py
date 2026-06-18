"""组装 LangGraph：planner → retrieval → tool → writer → critic →(retry?)。
langgraph 不可用时退化为等价的顺序+循环执行，保证任何环境可演示。"""
from __future__ import annotations

from app.config import settings
from app.graph.state import AgentState
from app.graph.nodes import (
    planner_node, retrieval_node, tool_node, writer_node, critic_node, should_retry,
)


def _build_compiled():
    from langgraph.graph import StateGraph, END

    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("tool", tool_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)
    g.set_entry_point("planner")
    g.add_edge("planner", "retrieval")
    g.add_edge("retrieval", "tool")
    g.add_edge("tool", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", should_retry, {"retry": "retrieval", "end": END})
    return g.compile()


_compiled = None


def _run_sequential(query: str) -> AgentState:
    state: AgentState = {"query": query, "trace": [], "iterations": 0}
    state.update(planner_node(state))
    while True:
        state.update(retrieval_node(state))
        state.update(tool_node(state))
        state.update(writer_node(state))
        state.update(critic_node(state))
        if should_retry(state) == "end":
            break
    return state


def run_query(query: str) -> AgentState:
    global _compiled
    try:
        if _compiled is None:
            _compiled = _build_compiled()
        return _compiled.invoke({"query": query, "trace": [], "iterations": 0})
    except Exception:
        return _run_sequential(query)
