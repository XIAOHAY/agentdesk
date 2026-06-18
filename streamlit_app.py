"""AgentDesk - Streamlit 可视化仪表盘

把 LangGraph 编排的 Agentic RAG 全流程可视化：查询改写 -> 混合检索(向量+BM25+Rerank)
-> 工具调用 -> 带证据生成 -> Critic 反思重试。直接在进程内调用 app.graph.run_query，
无需单独起 FastAPI；无 OPENAI_API_KEY 也能跑（离线 fallback）。

部署：Streamlit Community Cloud，入口文件 = agentdesk/streamlit_app.py。
"""
from __future__ import annotations

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st


def _load_secrets_into_env() -> None:
    """Secrets -> 环境变量（必须在 import app.* 之前，让 config 读到 key）。"""
    keys = [
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "CHAT_MODEL", "EMBEDDING_MODEL",
        "TOP_K", "MAX_ITERATIONS",
    ]
    for k in keys:
        try:
            if k in st.secrets and str(st.secrets[k]).strip():
                os.environ.setdefault(k, str(st.secrets[k]))
        except Exception:
            pass


_load_secrets_into_env()

from app.config import settings  # noqa: E402
from app.rag.indexer import build_index, INDEX_PATH  # noqa: E402
from app.graph.build_graph import run_query  # noqa: E402

st.set_page_config(
    page_title="AgentDesk - Agentic RAG 仪表盘",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1180px;}
      .ad-hero {background: linear-gradient(120deg,#1e3a8a 0%,#2563eb 55%,#7c3aed 100%);
        color:#fff; padding:1.3rem 1.6rem; border-radius:16px; margin-bottom:1.1rem;}
      .ad-hero h1 {margin:0; font-size:1.5rem; letter-spacing:.5px;}
      .ad-hero p {margin:.35rem 0 0; opacity:.92; font-size:.9rem;}
      .ad-card {border:1px solid #e6e8ef; border-radius:12px; padding:.85rem 1rem;
        margin-bottom:.65rem; background:#fff;}
      .ad-chunk {font-family:ui-monospace,Menlo,monospace; font-size:.78rem; color:#2563eb; font-weight:600;}
      .ad-bar {height:7px; border-radius:6px; background:#eef1f7; overflow:hidden; margin:.35rem 0 .15rem;}
      .ad-bar > span {display:block; height:100%; background:linear-gradient(90deg,#2563eb,#7c3aed);}
      .ad-pill {display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.72rem; font-weight:600;}
      .ad-step {border-left:3px solid #2563eb; padding:.2rem 0 .55rem .8rem; margin-left:.35rem; position:relative;}
      .ad-step:before {content:""; position:absolute; left:-7px; top:.35rem; width:11px; height:11px;
        border-radius:50%; background:#2563eb; border:2px solid #fff;}
      .ad-step small {color:#64748b;}
      .ad-muted {color:#64748b; font-size:.82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="首次启动：正在构建知识库索引...")
def ensure_index() -> int:
    # 冷启动自举：缺生成语料则用固定 seed 确定性重建（Cloud 不入库 data/docs 生成物时也能跑）
    docs_dir = os.path.join("data", "docs")
    if not os.path.exists(os.path.join(docs_dir, "plan_AC-100.md")):
        try:
            os.makedirs("eval", exist_ok=True)
            from scripts.gen_corpus import gen
            gen()
        except Exception:
            pass
    if not os.path.exists(INDEX_PATH):
        store = build_index()
        return len(store)
    try:
        import json
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return len(data.get("chunks", []))
        return -1
    except Exception:
        return -1


n_chunks = ensure_index()

NODE_LABELS = {
    "planner": "1) Planner - 查询改写",
    "retrieval": "2) Retrieval - 混合检索 + Rerank",
    "tool": "3) Tool - 工具路由",
    "writer": "4) Writer - 带证据生成",
    "critic": "5) Critic - faithfulness 反思",
}

SAMPLES = [
    "公司A和公司B 2025年营收分别是多少？",
    "知识库里有多少个文档？",
    "(210-205)/205*100",
    "AC-104 这个需求计划讲了什么？",
    "公司的报销政策是怎样的？",
]

with st.sidebar:
    st.markdown("### 运行配置")
    mode = "真实大模型" if settings.use_llm else "离线 fallback"
    st.markdown(f"**模型模式**：{mode}")
    if not settings.use_llm:
        st.caption("未配置 OPENAI_API_KEY，使用哈希向量 + 拼接答案。"
                   "在 Cloud 的 Secrets 填 key 即可启用真实大模型。")
    st.markdown(
        f"- 向量后端：`{settings.vector_backend}`\n"
        f"- Top-K：`{settings.top_k}`，最大反思轮数：`{settings.max_iterations}`\n"
        f"- 知识库 chunk 数：`{n_chunks}`"
    )
    st.divider()
    st.markdown("### 示例问题")
    for q in SAMPLES:
        if st.button(q, use_container_width=True, key=f"s_{q}"):
            st.session_state["pending_q"] = q
    st.divider()
    with st.expander("系统架构 / 流程"):
        st.markdown(
            "**编排（LangGraph）**：planner -> retrieval -> tool -> writer -> critic，"
            "critic 判定不忠实且未超轮数则回到 retrieval 重试。\n\n"
            "**检索**：多查询改写 -> 向量召回 + BM25 -> RRF 融合 -> Rerank。\n\n"
            "**工具层**：MCP 风格 registry（计算器走 AST 白名单 / kb_stats）。\n\n"
            "**可靠性**：faithfulness 评判 + 重试循环；langgraph 不可用时顺序兜底。"
        )

st.markdown(
    """
    <div class="ad-hero">
      <h1>AgentDesk - Agentic RAG 多智能体仪表盘</h1>
      <p>LangGraph 编排 - 混合检索(向量+BM25+Rerank) - MCP 工具层 - Critic 反思重试，全流程可视化</p>
    </div>
    """,
    unsafe_allow_html=True,
)

default_q = st.session_state.pop("pending_q", "")
query = st.text_input(
    "向知识库提问",
    value=default_q,
    placeholder="例如：公司A和公司B 2025年营收分别是多少？",
)
go = st.button("运行 Agent 流程", type="primary")

if (go or default_q) and query.strip():
    with st.spinner("Agent 编排执行中：改写 -> 检索 -> 工具 -> 生成 -> 反思..."):
        state = run_query(query.strip())

    answer = state.get("answer", "")
    verify = state.get("verify", {}) or {}
    iterations = state.get("iterations", 0)
    evidence = state.get("evidence", []) or []
    tool_results = state.get("tool_results", []) or []
    trace = state.get("trace", []) or []

    score = float(verify.get("score", 0) or 0)
    faithful = bool(verify.get("faithful"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Faithfulness", f"{score:.2f}", "通过" if faithful else "未达标")
    m2.metric("反思轮数", iterations)
    m3.metric("命中证据", len(evidence))
    m4.metric("评判方式", verify.get("method", "-"))

    col_main, col_side = st.columns([1.35, 1])

    with col_main:
        st.markdown("#### 答案")
        st.markdown(f"<div class='ad-card'>{answer or '（无答案）'}</div>", unsafe_allow_html=True)

        cites = state.get("citations", []) or []
        if cites:
            st.markdown("#### 引用")
            st.markdown(
                " ".join(
                    f"<span class='ad-pill' style='background:#eef2ff;color:#4338ca'>{c}</span>"
                    for c in cites
                ),
                unsafe_allow_html=True,
            )

        if tool_results:
            st.markdown("#### 工具调用")
            for r in tool_results:
                out = r.get("out", {})
                ok = out.get("ok")
                via = out.get("via", "local")
                color = "#dcfce7" if ok else "#fee2e2"
                txt = out.get("result", out.get("error", ""))
                st.markdown(
                    f"<div class='ad-card'><span class='ad-pill' style='background:{color}'>"
                    f"{r.get('tool')} - via {via}</span>"
                    f"<div style='margin-top:.4rem'><code>{txt}</code></div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("#### 检索证据")
        from dataclasses import asdict, is_dataclass
        for e in evidence:
            ev = asdict(e) if is_dataclass(e) else e
            sc = float(ev.get("score", 0) or 0)
            pct = max(4, min(100, int(sc * 100)))
            text = (ev.get("text", "") or "")[:240]
            st.markdown(
                f"<div class='ad-card'>"
                f"<span class='ad-chunk'>{ev.get('chunk_id')}</span> "
                f"<span class='ad-muted'>- {ev.get('doc_id')} - score {sc:.4f}</span>"
                f"<div class='ad-bar'><span style='width:{pct}%'></span></div>"
                f"<div class='ad-muted' style='margin-top:.3rem'>{text}...</div></div>",
                unsafe_allow_html=True,
            )

    with col_side:
        st.markdown("#### 执行链 Trace")
        for step in trace:
            node = step.get("node", "?")
            label = NODE_LABELS.get(node, node)
            parts = []
            if node == "planner":
                parts.append("改写: " + " | ".join(step.get("queries", [])))
            elif node == "retrieval":
                hits = step.get("hits", [])
                parts.append(f"iter {step.get('iter')} - {step.get('mode')} - {len(hits)} 命中")
            elif node == "tool":
                called = step.get("called", [])
                parts.append("调用: " + (", ".join(called) if called else "无"))
            elif node == "writer":
                parts.append(f"引用 {len(step.get('citations', []))} 条")
            elif node == "critic":
                parts.append(f"faithful={step.get('faithful')} - score={step.get('score')}")
            detail = "<br/>".join(parts)
            st.markdown(
                f"<div class='ad-step'><strong>{label}</strong><br/><small>{detail}</small></div>",
                unsafe_allow_html=True,
            )

        with st.expander("原始 state（调试）"):
            st.json({
                "iterations": iterations,
                "verify": verify,
                "citations": state.get("citations", []),
                "trace": trace,
            })

else:
    st.info("在上方输入问题，或从侧边栏点选示例问题，开始运行 Agent 流程。"
            "无需 API key 也能体验完整链路（离线 fallback）。")
