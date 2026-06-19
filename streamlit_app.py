"""AgentDesk - Agentic RAG 控制台（Streamlit）

顶尖视觉重设计版：深色 AI 控制台主题 + Design Token + 玻璃拟态卡片 + 微交互。
进程内直接调用 app.graph.run_query；无 OPENAI_API_KEY 也能跑（离线 fallback）。
部署：Streamlit Community Cloud，入口 = agentdesk/streamlit_app.py。
"""
from __future__ import annotations

import html
import os
from dataclasses import asdict, is_dataclass

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st


def _load_secrets_into_env() -> None:
    keys = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "CHAT_MODEL", "EMBEDDING_MODEL",
            "TOP_K", "MAX_ITERATIONS"]
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

st.set_page_config(page_title="AgentDesk · Agentic RAG 控制台",
                   page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ============================ Design System (CSS) ============================
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

      :root{
        --bg:#0a0e1a; --bg2:#0f1424; --surface:rgba(255,255,255,.045);
        --surface-2:rgba(255,255,255,.07); --stroke:rgba(255,255,255,.10);
        --stroke-2:rgba(255,255,255,.16);
        --ink:#eef2ff; --muted:#9aa6c4; --faint:#6b7596;
        --brand:#7c5cff; --brand2:#22d3ee; --accent:#f472b6;
        --ok:#34d399; --warn:#fbbf24; --bad:#fb7185;
        --r-s:10px; --r-m:16px; --r-l:22px;
        --grad:linear-gradient(120deg,#7c5cff 0%,#5b8cff 45%,#22d3ee 100%);
      }

      /* —— 背景：深空 + 双径向光晕 + 细网格 —— */
      .stApp{
        background:
          radial-gradient(1100px 620px at 12% -8%, rgba(124,92,255,.22), transparent 60%),
          radial-gradient(900px 560px at 105% 8%, rgba(34,211,238,.16), transparent 55%),
          linear-gradient(180deg,#0a0e1a 0%, #0b1020 60%, #0a0e1a 100%);
        color:var(--ink);
        font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
      }
      .block-container{padding-top:2.0rem; padding-bottom:3rem; max-width:1240px;}
      .stApp:before{
        content:""; position:fixed; inset:0; pointer-events:none; opacity:.5; z-index:0;
        background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),
          linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
        background-size:46px 46px; mask-image:radial-gradient(circle at 50% 0%,#000,transparent 75%);
      }
      h1,h2,h3,h4,p,span,div,label{font-family:'Inter',sans-serif;}
      a{color:var(--brand2);}
      ::selection{background:rgba(124,92,255,.35);}

      /* —— Hero —— */
      .hero{position:relative; overflow:hidden; border:1px solid var(--stroke);
        border-radius:var(--r-l); padding:30px 32px; margin-bottom:20px;
        background:linear-gradient(135deg, rgba(124,92,255,.20), rgba(34,211,238,.10) 55%, rgba(255,255,255,.02));
        box-shadow:0 30px 80px -40px rgba(91,140,255,.55), inset 0 1px 0 rgba(255,255,255,.08);}
      .hero:after{content:""; position:absolute; width:340px; height:340px; right:-90px; top:-150px;
        background:conic-gradient(from 120deg,#7c5cff,#22d3ee,#f472b6,#7c5cff); filter:blur(60px);
        opacity:.30; border-radius:50%; animation:spin 18s linear infinite;}
      @keyframes spin{to{transform:rotate(360deg);}}
      .hero h1{margin:0; font-size:2.0rem; font-weight:800; letter-spacing:-.5px;
        background:linear-gradient(90deg,#fff,#cdd6ff 60%,#9fe9ff); -webkit-background-clip:text;
        background-clip:text; -webkit-text-fill-color:transparent;}
      .hero p{margin:.5rem 0 0; color:#c5cdf0; font-size:.96rem; max-width:760px; line-height:1.5;}
      .hero .chips{margin-top:16px; display:flex; gap:8px; flex-wrap:wrap;}
      .chip{display:inline-flex; align-items:center; gap:7px; padding:6px 13px; border-radius:999px;
        font-size:.78rem; font-weight:600; border:1px solid var(--stroke-2);
        background:var(--surface-2); color:#d7dcf5; backdrop-filter:blur(8px);}
      .dot{width:8px;height:8px;border-radius:50%;box-shadow:0 0 12px currentColor;}

      /* —— 区块标题 —— */
      .eyebrow{display:flex; align-items:center; gap:9px; margin:6px 0 12px;
        font-size:.74rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:var(--faint);}
      .eyebrow:before{content:""; width:18px; height:2px; border-radius:2px; background:var(--grad);}

      /* —— 通用卡片 —— */
      .card{position:relative; border:1px solid var(--stroke); border-radius:var(--r-m);
        background:var(--surface); backdrop-filter:blur(10px); padding:16px 18px; margin-bottom:14px;
        box-shadow:0 18px 40px -30px rgba(0,0,0,.8), inset 0 1px 0 rgba(255,255,255,.05);
        transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;}
      .card:hover{transform:translateY(-2px); border-color:var(--stroke-2);
        box-shadow:0 26px 60px -34px rgba(91,140,255,.5), inset 0 1px 0 rgba(255,255,255,.07);}

      /* —— KPI —— */
      .kpi{border:1px solid var(--stroke); border-radius:var(--r-m); padding:16px 18px; height:100%;
        background:linear-gradient(180deg,var(--surface-2),var(--surface)); position:relative; overflow:hidden;}
      .kpi .k-ico{font-size:1.05rem; opacity:.95;}
      .kpi .k-lab{color:var(--muted); font-size:.76rem; font-weight:600; letter-spacing:.04em; margin-top:6px;}
      .kpi .k-val{font-size:1.95rem; font-weight:800; letter-spacing:-.5px; line-height:1.1; margin-top:2px;
        font-variant-numeric:tabular-nums;}
      .kpi .k-sub{font-size:.74rem; color:var(--faint); margin-top:3px;}
      .kpi:after{content:""; position:absolute; left:0; bottom:0; height:3px; width:100%; background:var(--grad); opacity:.85;}

      /* —— Faithfulness 环形仪表 —— */
      .gauge-wrap{display:flex; align-items:center; gap:16px;}
      .gauge{--p:0; width:92px; height:92px; border-radius:50%; flex:0 0 auto; position:relative;
        background:conic-gradient(var(--gc,#34d399) calc(var(--p)*1%), rgba(255,255,255,.08) 0);
        display:grid; place-items:center; box-shadow:0 0 0 1px var(--stroke) inset;}
      .gauge:before{content:""; position:absolute; inset:9px; border-radius:50%; background:#0c1122;
        box-shadow:inset 0 1px 0 rgba(255,255,255,.06);}
      .gauge b{position:relative; font-size:1.25rem; font-weight:800; font-variant-numeric:tabular-nums;}

      /* —— 引用 / pill —— */
      .pill{display:inline-flex; align-items:center; gap:6px; padding:5px 11px; border-radius:999px;
        font-size:.74rem; font-weight:600; margin:0 6px 6px 0; border:1px solid var(--stroke-2);
        background:rgba(124,92,255,.14); color:#d9d2ff; font-family:'JetBrains Mono',monospace;}
      .pill.tool{background:rgba(52,211,153,.14); color:#b8f5dd;}
      .pill.bad{background:rgba(251,113,133,.14); color:#ffc6cf;}

      /* —— 证据卡 —— */
      .ev{border:1px solid var(--stroke); border-radius:var(--r-m); padding:14px 16px; margin-bottom:12px;
        background:var(--surface); transition:transform .16s ease,border-color .16s ease;}
      .ev:hover{transform:translateX(3px); border-color:var(--stroke-2);}
      .ev-top{display:flex; align-items:center; justify-content:space-between; gap:10px;}
      .ev-id{font-family:'JetBrains Mono',monospace; font-size:.8rem; font-weight:600; color:#a9b6ff;}
      .ev-sc{font-family:'JetBrains Mono',monospace; font-size:.74rem; color:var(--muted);}
      .ev-rank{width:22px;height:22px;border-radius:7px;display:grid;place-items:center;font-size:.72rem;
        font-weight:700; color:#0a0e1a; background:var(--grad); flex:0 0 auto;}
      .bar{height:6px; border-radius:6px; background:rgba(255,255,255,.07); overflow:hidden; margin:10px 0 8px;}
      .bar>span{display:block; height:100%; border-radius:6px; background:var(--grad);
        box-shadow:0 0 14px rgba(124,92,255,.6); animation:grow .6s cubic-bezier(.2,.8,.2,1);}
      @keyframes grow{from{width:0;}}
      .ev-txt{color:#c3cbe6; font-size:.84rem; line-height:1.55;}
      .ans{color:#e9edff; font-size:.95rem; line-height:1.7; white-space:pre-wrap;}

      /* —— 流程时间线 —— */
      .tl{position:relative; margin-left:6px; padding-left:22px;}
      .tl:before{content:""; position:absolute; left:5px; top:6px; bottom:6px; width:2px;
        background:linear-gradient(180deg,#7c5cff,#22d3ee);}
      .node{position:relative; padding:0 0 16px 4px;}
      .node:before{content:""; position:absolute; left:-22px; top:3px; width:13px; height:13px; border-radius:50%;
        background:#0a0e1a; border:2px solid #7c5cff; box-shadow:0 0 0 4px rgba(124,92,255,.12);}
      .node.done:before{background:var(--grad); border-color:transparent;}
      .node .n-t{font-size:.86rem; font-weight:700; color:#e7ebff;}
      .node .n-d{font-size:.78rem; color:var(--muted); margin-top:3px; line-height:1.5;}

      /* —— Streamlit 控件覆写 —— */
      section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b0f1e,#0a0e1a);
        border-right:1px solid var(--stroke);}
      section[data-testid="stSidebar"] .block-container{padding-top:1.4rem;}
      .stTextInput input{background:var(--surface-2)!important; color:var(--ink)!important;
        border:1px solid var(--stroke-2)!important; border-radius:14px!important; height:52px; font-size:.95rem;
        padding:0 16px!important;}
      .stTextInput input::placeholder{color:var(--faint)!important;}
      .stTextInput input:focus{border-color:var(--brand)!important;
        box-shadow:0 0 0 3px rgba(124,92,255,.25)!important;}
      .stTextInput label{color:var(--muted)!important; font-weight:600!important;}
      .stButton>button{border-radius:13px; border:1px solid var(--stroke-2); font-weight:600;
        background:var(--surface-2); color:#dfe4ff; transition:all .16s ease;}
      .stButton>button:hover{border-color:var(--brand); color:#fff; transform:translateY(-1px);
        background:rgba(124,92,255,.16);}
      .stButton>button[kind="primary"]{background:var(--grad); border:none; color:#fff; height:50px;
        font-weight:700; letter-spacing:.02em; box-shadow:0 14px 34px -14px rgba(124,92,255,.85);}
      .stButton>button[kind="primary"]:hover{filter:brightness(1.08); transform:translateY(-1px);}
      div[data-testid="stExpander"]{border:1px solid var(--stroke)!important; border-radius:14px!important;
        background:var(--surface)!important; overflow:hidden;}
      div[data-testid="stExpander"] summary{color:#cdd5f5!important; font-weight:600!important;}
      hr{border-color:var(--stroke)!important;}
      #MainMenu,header[data-testid="stHeader"],footer{visibility:hidden;}
      .stApp > div{z-index:1;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================ 启动：建索引（自举 + 缓存） ============================
_META_PATH = os.path.join(os.path.dirname(INDEX_PATH), "index_meta.json")


def _emb_signature() -> dict:
    # 索引指纹：embedding 维度由「是否真实模型 + 模型名」决定，变了就必须重建
    return {"use_llm": bool(settings.use_llm),
            "model": settings.embedding_model if settings.use_llm else "offline-hash"}


def _read_meta():
    try:
        import json
        with open(_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _rebuild_index() -> int:
    import json
    # 清掉旧维度的 embedding 缓存与检索器单例，确保按当前维度重建（离线 256 ↔ 真实 1024）
    try:
        from app.rag.cache import cache as _c
        getattr(_c, "_mem", {}).clear()
    except Exception:
        pass
    try:
        import app.graph.nodes as _n
        _n._retriever = None
    except Exception:
        pass
    n = len(build_index())
    try:
        os.makedirs(os.path.dirname(_META_PATH), exist_ok=True)
        with open(_META_PATH, "w", encoding="utf-8") as f:
            json.dump(_emb_signature(), f)
    except Exception:
        pass
    return n


@st.cache_resource(show_spinner="冷启动：正在构建知识库索引…")
def ensure_index() -> int:
    docs_dir = os.path.join("data", "docs")
    if not os.path.exists(os.path.join(docs_dir, "plan_AC-100.md")):
        try:
            os.makedirs("eval", exist_ok=True)
            from scripts.gen_corpus import gen
            gen()
        except Exception:
            pass
    # 无索引，或 embedding 指纹变了（离线↔真实模型切换导致维度不匹配）→ 重建
    if not os.path.exists(INDEX_PATH) or _read_meta() != _emb_signature():
        return _rebuild_index()
    try:
        import json
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else len(data.get("chunks", []))
    except Exception:
        return -1


n_chunks = ensure_index()

NODE_LABELS = {
    "planner": ("Planner", "查询改写 · multi-query"),
    "retrieval": ("Retrieval", "向量 + BM25 → RRF → Rerank"),
    "tool": ("Tool", "MCP 工具路由"),
    "writer": ("Writer", "带证据生成 · 标注引用"),
    "critic": ("Critic", "faithfulness 反思判定"),
}
SAMPLES = [
    "公司A和公司B 2025年营收分别是多少？",
    "知识库里有多少个文档？",
    "(210-205)/205*100",
    "AC-104 这个需求计划讲了什么？",
    "公司的报销政策是怎样的？",
]


def esc(x) -> str:
    return html.escape(str(x))


# ============================ 侧边栏 ============================
with st.sidebar:
    st.markdown("<div class='eyebrow'>Console</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.15rem;font-weight:800;margin:-4px 0 2px'>AgentDesk</div>"
                "<div style='color:var(--muted);font-size:.8rem'>Agentic RAG · 多智能体</div>",
                unsafe_allow_html=True)
    st.markdown("<hr style='margin:14px 0'>", unsafe_allow_html=True)

    live = settings.use_llm
    st.markdown(
        f"<div class='card' style='margin-bottom:12px'>"
        f"<div style='display:flex;align-items:center;gap:9px'>"
        f"<span class='dot' style='color:{'#34d399' if live else '#fbbf24'}'></span>"
        f"<b style='font-size:.9rem'>{'真实大模型' if live else '离线 Fallback'}</b></div>"
        f"<div style='color:var(--faint);font-size:.76rem;margin-top:6px;line-height:1.5'>"
        f"{'已接入 LLM/Embedding API' if live else '哈希向量 + 拼接答案，无需任何 key'}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>"
        f"<div class='card' style='margin:0;padding:12px 14px'><div style='color:var(--faint);font-size:.7rem'>向量后端</div>"
        f"<div style='font-weight:700;margin-top:3px'>{esc(settings.vector_backend)}</div></div>"
        f"<div class='card' style='margin:0;padding:12px 14px'><div style='color:var(--faint);font-size:.7rem'>Top-K</div>"
        f"<div style='font-weight:700;margin-top:3px'>{esc(settings.top_k)}</div></div>"
        f"<div class='card' style='margin:0;padding:12px 14px'><div style='color:var(--faint);font-size:.7rem'>反思上限</div>"
        f"<div style='font-weight:700;margin-top:3px'>{esc(settings.max_iterations)}</div></div>"
        f"<div class='card' style='margin:0;padding:12px 14px'><div style='color:var(--faint);font-size:.7rem'>KB chunks</div>"
        f"<div style='font-weight:700;margin-top:3px'>{esc(n_chunks)}</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='eyebrow' style='margin-top:18px'>试一试</div>", unsafe_allow_html=True)
    for q in SAMPLES:
        if st.button(q, use_container_width=True, key=f"s_{q}"):
            st.session_state["qbox"] = q
            st.session_state["_autorun"] = True
    with st.expander("架构 / 流程"):
        st.markdown(
            "**编排（LangGraph）**：planner → retrieval → tool → writer → critic；"
            "critic 不达标且未超轮数则回 retrieval 重试。\n\n"
            "**检索**：多查询改写 → 向量 + BM25 → RRF 融合 → Rerank。\n\n"
            "**工具层**：MCP 风格 registry（AST 白名单计算器 / kb_stats）。\n\n"
            "**兜底**：langgraph 不可用时顺序等价执行。"
        )

# ============================ Hero ============================
st.markdown(
    f"""
    <div class="hero">
      <h1>Agentic RAG 控制台</h1>
      <p>LangGraph 编排的多智能体检索增强系统 · 把<b>查询改写 → 混合检索 → 工具调用 → 带证据生成 → 反思重试</b>的全过程实时可视化。</p>
      <div class="chips">
        <span class="chip"><span class="dot" style="color:{'#34d399' if settings.use_llm else '#fbbf24'}"></span>{'真实大模型' if settings.use_llm else '离线 Fallback'}</span>
        <span class="chip">🧩 混合检索 向量+BM25+Rerank</span>
        <span class="chip">🛠️ MCP 工具层</span>
        <span class="chip">🔁 Critic 反思循环</span>
        <span class="chip">📚 {esc(n_chunks)} chunks</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================ 提问区 ============================
st.markdown("<div class='eyebrow'>Ask the knowledge base</div>", unsafe_allow_html=True)

# —— Chat 模型选择（放主区，醒目）——
# 运行时覆盖 settings.chat_model；app/llm.py:chat() 每次现读该值，
# 故一个开关同时作用于 改写(planner)/生成(writer)/裁判(critic) 三处，无需改图或穿参。
_live = settings.use_llm
_PRESETS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-72B-Instruct",
            "deepseek-ai/DeepSeek-V3", "自定义…"]
_cur = st.session_state.get("chat_model", settings.chat_model)
_opts = _PRESETS if _cur in _PRESETS else [_cur] + _PRESETS
m_lab, m_sel, m_cus = st.columns([1, 2, 2])
with m_lab:
    st.markdown("<div style='padding-top:10px;color:var(--muted);font-weight:600;font-size:.9rem'>🧠 Chat 模型</div>",
                unsafe_allow_html=True)
with m_sel:
    _pick = st.selectbox("chat 模型", _opts,
                         index=_opts.index(_cur) if _cur in _opts else 0,
                         label_visibility="collapsed", disabled=not _live)
with m_cus:
    if _pick == "自定义…":
        _pick = st.text_input("自定义模型名", value=("" if _cur in _PRESETS else _cur),
                              placeholder="如 Qwen/Qwen2.5-72B-Instruct",
                              label_visibility="collapsed", disabled=not _live).strip()
    else:
        _hint = ("影响 改写/生成/裁判三处 · 7B 易把数字写崩，建议 32B+" if _live
                 else "离线 fallback 不调用大模型，切换无效（需在 .env 配 key）")
        st.markdown(f"<div style='padding-top:11px;color:var(--faint);font-size:.74rem'>{_hint}</div>",
                    unsafe_allow_html=True)
_pick = _pick or settings.chat_model
if _live:
    settings.chat_model = _pick               # chat() 每次现读 → 立即生效
    os.environ["CHAT_MODEL"] = _pick
st.session_state["chat_model"] = settings.chat_model

# —— 示例问题（点击即问；放输入框上方，省去面试官打字）——
# 置于输入行之前：按钮在本次 rerun 先于下方 pop("pending_q") 执行，故点击当次即填入并运行。
_DEMOS = [
    ("📊 多公司营收", "公司A和公司B 2025年营收分别是多少？"),
    ("🔢 计算器工具", "(210-205)/205*100"),
    ("🗂️ 知识库统计", "知识库里有多少个文档？"),
    ("📄 套餐 SLA", "AC-110 套餐的 SLA 可用性是多少？"),
    ("📑 报销政策", "公司的报销政策是怎样的？"),
]
st.markdown("<div style='color:var(--faint);font-size:.75rem;margin:2px 0 7px'>示例 · 点击即问</div>",
            unsafe_allow_html=True)
_dc = st.columns(len(_DEMOS))
for _i, (_lab, _q) in enumerate(_DEMOS):
    if _dc[_i].button(_lab, key=f"demo_{_i}", use_container_width=True):
        st.session_state["qbox"] = _q          # 直接写入输入框的 state（key 绑定）
        st.session_state["_autorun"] = True    # 标记：本次点击需自动运行一次

c_in, c_btn = st.columns([4, 1])
with c_in:
    # 用 key 绑定 session_state['qbox']；示例/侧边栏按钮已在上方写好它，
    # 故无需 value=（value= 在按钮场景下常不回显，正是“点了没反应”的根因）。
    query = st.text_input("向知识库提问", key="qbox", label_visibility="collapsed",
                          placeholder="例如：公司A和公司B 2025年营收分别是多少？")
with c_btn:
    go = st.button("⚡ 运行", type="primary", use_container_width=True)

_autorun = st.session_state.pop("_autorun", False)
# ============================ 运行与渲染 ============================
if (go or _autorun) and query.strip():
    with st.spinner("Agent 编排执行中：改写 → 检索 → 工具 → 生成 → 反思…"):
        state = run_query(query.strip())

    answer = state.get("answer", "")
    verify = state.get("verify", {}) or {}
    iterations = state.get("iterations", 0)
    evidence = state.get("evidence", []) or []
    tool_results = state.get("tool_results", []) or []
    trace = state.get("trace", []) or []
    cites = state.get("citations", []) or []

    score = float(verify.get("score", 0) or 0)
    faithful = bool(verify.get("faithful"))
    pct = max(0, min(100, int(round(score * 100))))
    gc = "#34d399" if faithful else ("#fbbf24" if score >= 0.4 else "#fb7185")

    # —— KPI 行 —— 
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, "🎯", "Faithfulness", f"{score:.2f}", ("证据支撑达标" if faithful else "未达标")),
        (k2, "🔁", "反思轮数", f"{iterations}", "critic retry loop"),
        (k3, "📎", "命中证据", f"{len(evidence)}", "RRF + Rerank top-k"),
        (k4, "⚖️", "评判方式", f"{verify.get('method','-')}", "LLM-judge / 启发式"),
    ]
    for col, ico, lab, val, sub in kpis:
        col.markdown(
            f"<div class='kpi'><div class='k-ico'>{ico}</div><div class='k-lab'>{lab}</div>"
            f"<div class='k-val'>{esc(val)}</div><div class='k-sub'>{esc(sub)}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    main, side = st.columns([1.4, 1], gap="large")

    with main:
        st.markdown("<div class='eyebrow'>Answer</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='card'><div class='gauge-wrap'>"
            f"<div class='gauge' style='--p:{pct};--gc:{gc}'><b>{pct}%</b></div>"
            f"<div><div style='font-weight:700;color:#fff'>{'✅ 可信回答' if faithful else '⚠️ 证据支撑不足'}</div>"
            f"<div style='color:var(--muted);font-size:.8rem;margin-top:4px'>faithfulness = 答案被检索证据支撑的比例</div></div>"
            f"</div><div class='ans' style='margin-top:14px'>{esc(answer) or '（无答案）'}</div></div>",
            unsafe_allow_html=True,
        )

        if cites:
            st.markdown("<div class='eyebrow'>Citations</div>", unsafe_allow_html=True)
            st.markdown("".join(f"<span class='pill'>🔖 {esc(c)}</span>" for c in cites),
                        unsafe_allow_html=True)

        if tool_results:
            st.markdown("<div class='eyebrow'>Tool calls</div>", unsafe_allow_html=True)
            for r in tool_results:
                out = r.get("out", {}) or {}
                ok = out.get("ok")
                cls = "tool" if ok else "bad"
                txt = out.get("result", out.get("error", ""))
                st.markdown(
                    f"<div class='card' style='padding:13px 16px'>"
                    f"<span class='pill {cls}'>{'✓' if ok else '✕'} {esc(r.get('tool'))} · via {esc(out.get('via','local'))}</span>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.84rem;color:#dfe6ff;margin-top:8px'>{esc(txt)}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='eyebrow'>Retrieved evidence</div>", unsafe_allow_html=True)
        for i, e in enumerate(evidence, 1):
            ev = asdict(e) if is_dataclass(e) else e
            sc = float(ev.get("score", 0) or 0)
            w = max(5, min(100, int(sc * 100)))
            txt = (ev.get("text", "") or "")[:240]
            st.markdown(
                f"<div class='ev'><div class='ev-top'>"
                f"<div style='display:flex;align-items:center;gap:10px'><span class='ev-rank'>{i}</span>"
                f"<span class='ev-id'>{esc(ev.get('chunk_id'))}</span></div>"
                f"<span class='ev-sc'>{esc(ev.get('doc_id'))} · {sc:.4f}</span></div>"
                f"<div class='bar'><span style='width:{w}%'></span></div>"
                f"<div class='ev-txt'>{esc(txt)}…</div></div>",
                unsafe_allow_html=True,
            )

    with side:
        st.markdown("<div class='eyebrow'>Execution trace</div>", unsafe_allow_html=True)
        nodes_html = "<div class='tl'>"
        for step in trace:
            node = step.get("node", "?")
            title, _sub = NODE_LABELS.get(node, (node, ""))
            if node == "planner":
                d = "改写 → " + esc(" / ".join(step.get("queries", [])))
            elif node == "retrieval":
                d = f"iter {esc(step.get('iter'))} · {esc(step.get('mode'))} · {len(step.get('hits', []))} 命中"
            elif node == "tool":
                called = step.get("called", [])
                d = "调用 " + (esc(", ".join(called)) if called else "（无）")
            elif node == "writer":
                d = f"生成答案 · 标注 {len(step.get('citations', []))} 条引用"
            elif node == "critic":
                d = f"faithful={esc(step.get('faithful'))} · score={esc(step.get('score'))}"
            else:
                d = ""
            nodes_html += (f"<div class='node done'><div class='n-t'>{esc(title)}</div>"
                           f"<div class='n-d'>{d}</div></div>")
        nodes_html += "</div>"
        st.markdown(nodes_html, unsafe_allow_html=True)

        with st.expander("原始 state（调试）"):
            st.json({"iterations": iterations, "verify": verify,
                     "citations": cites, "trace": trace})

else:
    st.markdown(
        "<div class='card' style='text-align:center;padding:40px 24px;border-style:dashed'>"
        "<div style='font-size:2rem'>🧠</div>"
        "<div style='font-weight:700;font-size:1.05rem;margin-top:8px'>输入问题，开始一次 Agent 编排</div>"
        "<div style='color:var(--muted);font-size:.86rem;margin-top:6px'>"
        "点上方示例问题，或直接提问 — 无需 API key 也能体验完整链路（离线 fallback）。</div></div>",
        unsafe_allow_html=True,
    )
