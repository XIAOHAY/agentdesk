# AgentDesk · Agentic RAG + 分层记忆 + 评测闭环

> 一个可复现的**企业知识问答 Agent 原型**：把 _记忆 → 检索 → 工具调用 → 带证据生成 → 反思重试 → 质量评测_ 串成一条可观测、可量化、可回归的闭环。

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
<img alt="LangGraph" src="https://img.shields.io/badge/Orchestration-LangGraph-7c5cff">
<img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
<img alt="Qdrant" src="https://img.shields.io/badge/Vector-Qdrant-dc244c">
<img alt="License" src="https://img.shields.io/badge/License-MIT-green">
<img alt="No API key needed" src="https://img.shields.io/badge/run-offline%20fallback-fbbf24">
</p>

**无需任何 API key 也能端到端运行**（自动走离线 fallback：哈希向量 + 拼接式回答）；Qdrant / Redis / 大模型 key 任一不可用都会自动回退内存/离线实现，本地零外部依赖即可演示完整链路。

---

## ✨ 核心特性

- **Agentic 编排（LangGraph）**：`memory_retrieve → planner → retrieval → tool → writer → critic →(重试)→ memory_write →(summarize)`，每个节点可观测；LangGraph 不可用时退化为等价顺序执行。
- **分层记忆（Memory Layer）**：短期工作记忆（对话 buffer + 滚动摘要）、长期记忆（偏好/事实抽取 → 向量化 → Qdrant 按 `user_id` namespace 隔离 → 检索注入）、记忆演化（写入去重 / 冲突覆盖留审计痕迹 / TTL+LRU 淘汰）。
- **混合检索 + 重排**：多查询改写 → 向量 + BM25 → RRF 融合 → Rerank。
- **可信回答与反思循环**：faithfulness 评估（有 key 用 LLM-as-judge，无 key 回退启发式），答案未被证据/工具支撑且未超迭代上限时自动回到检索重试；writer 出口净化无效/幻觉引用。
- **MCP 风格工具层**：`list_tools / call_tool` 契约 + 工具名/参数 schema 校验 + 输出截断；calculator 用 AST 白名单阻断注入。
- **评测闭环**：检索 `hit@k / MRR` + 记忆 `memory hit@k`，量化“检索好不好、记忆有没有被想起”。
- **全程可观测**：执行 trace 实时可视化，并落盘 `eval/reports/traces.jsonl` 供事后复盘。

---

## 🏗️ 架构

```
                        ┌──────────── LangGraph 编排 ────────────┐
 /chat (query,          │ memory_retrieve → planner → retrieval  │
  user_id, session_id)  │      → tool → writer → critic          │
        │               │            │(不达标且未超限则重试)        │
        ▼               │            └──► memory_write →(summarize)│
   FastAPI / Streamlit  └────────────────────────────────────────┘
        │                         │                    │
   实时 trace 可视化         Qdrant(知识库 + 记忆)      Redis(缓存 + 短期记忆)
                                  └─ 不可用→内存         └─ 不可用→内存
```

- 详细数据流见 [`docs/数据流与可观测.mermaid`](docs/数据流与可观测.mermaid)
- 记忆层设计见 [`docs/记忆层设计文档.md`](docs/记忆层设计文档.md) 与 [`docs/记忆数据流.mermaid`](docs/记忆数据流.mermaid)

---

## 🚀 快速开始

```bash
pip install -r requirements.txt

# （可选）配置真实模型；不配置则用离线 fallback
cp .env.example .env          # 填 OPENAI_API_KEY（或硅基流动/百炼等兼容厂商）

python -m scripts.build_index # 建索引
uvicorn app.api.main:app --reload
```

- Web 聊天界面：<http://localhost:8000/> · 交互式 API 文档：<http://localhost:8000/docs>
- 可视化仪表盘（推荐演示）：`streamlit run streamlit_app.py` —— 答案 + faithfulness 仪表盘 + 引用/证据 + 工具调用 + **对话历史 & 记忆面板 & 演化审计** + 执行链时间线。
- 一键全栈：`docker compose up --build`（api + qdrant + redis，均带回退）。

```bash
# 跨轮记忆演示（同一 user + session）
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"query":"我是法务，只看2024年的合规条款","user_id":"alice","session_id":"s1"}'
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"query":"这份合同我该重点看什么？","user_id":"alice","session_id":"s1"}'   # 第二轮会召回上面的记忆
```

---

## 🧠 分层记忆

| 层 | 做什么 | 存储 | 文件 |
|---|---|---|---|
| 短期工作记忆 | 对话 buffer + 滚动 summary 压缩（保留近 K 轮 + running_summary），控制长对话上下文膨胀 | Redis / 内存 | `app/memory/short_term.py` |
| 长期记忆 | 抽取用户偏好/事实 → 向量化 → Qdrant 按 `user_id` namespace 隔离 → 检索注入 | Qdrant / 内存 | `app/memory/long_term.py` |
| 记忆演化 | 写入去重（相似度阈值）、冲突更新（新值覆盖、旧值留 `version/superseded_by` 审计）、过期淘汰（event TTL + 容量 LRU） | — | `app/memory/evolution.py` |

入口/出口以 `memory_retrieve / memory_write / summarize` 三节点无侵入接入既有编排；阈值与开关见 `.env.example` 的 `MEM_*`。

---

## 📊 评测

```bash
python -m eval.run_eval            # 检索：vector / hybrid / hybrid+rerank 的 hit@k 与 MRR
python -m eval.run_memory_eval 5   # 记忆：memory hit@1/3/5 → eval/reports/memory_latest.json
python -m eval.run_faithfulness 8  # 生成侧：平均 faithfulness 与通过率
```

> 当前示例语料较小且离线 embedding 已接近天花板，hit@k 提升幅度有限；评测框架已就绪，换真实 embedding + 更大语料后混合检索与 Rerank 的提升会明显——这正是简历中 "X% → Y%" 的数据来源。

---

## 🔍 可观测

- 每个节点统一往 `trace` 追加结构化记录（改写 / 检索命中 / 工具调用 / faithfulness 分数 / 记忆读写），前端时间线实时渲染。
- 每轮执行链落盘 `eval/reports/traces.jsonl`（一行一条 JSON，便于 grep/回放）；`TRACE_LOG=0` 可关。

---

## 📁 目录

```
agentdesk/
├── app/
│   ├── config.py            # 配置（.env / 环境变量，全部带默认值）
│   ├── llm.py               # Embedding/Chat 封装（含离线 fallback + 缓存）
│   ├── rag/                 # store / indexer / retriever / bm25 / rerank / qdrant / cache
│   ├── memory/              # 分层记忆：schema / store / short_term / long_term / evolution
│   ├── graph/               # LangGraph: state / nodes / build_graph / judge / trace_log
│   ├── tools/               # MCP 风格工具层 + 内置工具 + stdio MCP server/client
│   └── api/main.py          # FastAPI: / · /chat · /health
├── eval/                    # 检索 hit@k/MRR · memory hit@k · faithfulness
├── scripts/                 # build_index / gen_corpus / demo / mcp_demo
├── docs/                    # 架构图 / 记忆层设计 / 数据流 mermaid
├── streamlit_app.py         # 可视化控制台
├── docker-compose.yml · Dockerfile · requirements.txt
```

---

## 🗺️ 里程碑

- [x] 朴素 RAG + FastAPI + LangGraph 编排
- [x] 查询改写 + 混合检索(向量+BM25, RRF) + Rerank + eval(hit@k/MRR)
- [x] tool 节点 + Critic 反思节点 + faithfulness 重试循环
- [x] MCP 风格工具层 + JSON-RPC over stdio MCP server + 安全计算器
- [x] Qdrant 向量库 + Redis 缓存 + docker-compose（均带回退）
- [x] LLM-judge faithfulness + 生成侧评测
- [x] **分层记忆（短期/长期/演化）+ memory hit@k + 引用净化 + trace 落盘**

---

## License

[MIT](LICENSE)
