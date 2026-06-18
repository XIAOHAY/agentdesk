# AgentDesk · 企业级 Agentic RAG + 多智能体系统

第 1 周里程碑脚手架：**LangGraph 编排 + FastAPI + 朴素 RAG**，已可端到端运行。
无需 API key 也能跑（自动走离线 fallback：哈希向量 + 拼接式回答），方便先验证骨架。

## 目录
```
agentdesk/
├── app/
│   ├── config.py          # 配置（.env）
│   ├── llm.py             # Embedding/Chat 封装（含离线 fallback）
│   ├── rag/               # store / indexer / retriever
│   ├── graph/             # LangGraph: state / nodes / build_graph
│   └── api/main.py        # FastAPI: /chat /health
├── scripts/build_index.py # 建索引
├── data/docs/             # 示例文档
├── requirements.txt
├── Dockerfile
└── README.md
```

## 快速开始
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2.（可选）配置真实模型；不配置则用离线 fallback
cp .env.example .env   # 填 OPENAI_API_KEY

# 3. 建索引
python -m scripts.build_index

# 4. 启动服务
uvicorn app.api.main:app --reload

# 5. 测试
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"公司A和公司B 2025年营收分别是多少？"}'
```

打开 **http://localhost:8000/** 是内置的 Web 聊天界面（输入框 + 答案 + 引用/证据/执行链展示）。
打开 http://localhost:8000/docs 可看交互式 API 文档。

## Streamlit 可视化仪表盘
除 FastAPI 外，另提供一个**演示级可视化界面**（进程内直接调用 `run_query`，无需起 API）：
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
界面把全流程可视化：答案 + faithfulness 仪表盘 + 引用 + 证据卡片(带 score 条) + 工具调用 +
执行链 Trace 时间线 + 侧边栏配置/架构。无 `OPENAI_API_KEY` 也能跑（离线 fallback）。

### 部署到 Streamlit Community Cloud（免费）
1. 代码推到 GitHub（public 仓库最省事）。
2. 打开 https://share.streamlit.io → **New app** → 选本仓库。
3. **Main file path** 填 `agentdesk/streamlit_app.py`（若仓库根就是 agentdesk，则填 `streamlit_app.py`）。
4. （可选）**Advanced settings → Secrets** 粘贴 `.streamlit/secrets.toml.example` 的内容并填真实 key，启用真实大模型；不填则用离线 fallback。
5. Deploy。首次启动会自动构建知识库索引（`data/index/` 不入库，运行时重建）。

> 内置语料生成器：`python -m scripts.gen_corpus` 会生成 20 篇相似主题文档 + 配套评测集，
> 专门设计成能体现混合检索优势（型号/数字精确匹配靠 BM25，语义改写靠向量）。

## 评测（量化检索提升）
```bash
python -m eval.run_eval
```
对比三种配置的 hit@k / MRR：`vector`（朴素向量）→ `hybrid`（向量+BM25 RRF 融合）→ `hybrid+rerank`。
评测在 doc 级别标注（`eval/dataset.jsonl`），对 chunk 切分鲁棒。

生成侧评测（faithfulness）：`python -m eval.run_faithfulness 8` —— 跑样本问题统计平均
faithfulness 与通过率；有 key 时 critic 用 **LLM-as-judge** 做事实核查，无 key 回退启发式。

> 注：离线 fallback 用哈希向量，且示例语料仅 4 篇、主题区分度高，朴素向量已接近天花板，
> 提升幅度看起来很小。**换成真实 embedding + 更大语料**后，混合检索与 Rerank 的提升会明显，
> 这才是简历里 “X% → Y%” 的真实数据来源。eval 框架本身已就绪、可直接复用。

## 工具层与反思（第3周）
- `app/tools/registry.py`：MCP 风格本地工具层。`list_tools()` 发现、`call_tool(name,args)` 调用，内置四道可靠性闸门——工具名校验、参数 schema 校验、输出截断、统一错误处理。后续把 `call_tool` 接到真正的 MCP Server（Stdio/HTTP）即可，契约不变。
- `app/tools/builtins.py`：`calculator`（AST 白名单，拒绝 `__import__` 等注入）、`kb_stats`。
- `critic_node`：判定答案对证据的 faithfulness，不达标且未超 `max_iterations` 则回 `retrieval` 重试（条件边 `should_retry`），有 LLM 时换 LLM-as-judge。

试一下：
```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{"query":"(210-205)/205*100"}'   # 触发 calculator
curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{"query":"知识库里有多少个文档？"}'  # 触发 kb_stats
```

## MCP 服务端（真实传输）
- `app/tools/mcp_server.py`：纯标准库实现的 **JSON-RPC 2.0 over stdio** MCP 服务端，方法 `initialize / tools/list / tools/call`，复用工具 registry。可换成官方 `mcp` SDK，契约一致。
- `app/tools/mcp_client.py`：spawn 服务端子进程，完成 initialize 握手并调用工具。
- `app/tools/dispatch.py`：`USE_MCP=1` 时工具调用走真实 MCP 子进程（返回 `via:"mcp"`），否则进程内 registry（`via:"local"`）。graph 无感知。

演示真实传输：
```bash
python -m scripts.mcp_demo          # 握手 + tools/list + tools/call + 错误回传
USE_MCP=1 uvicorn app.api.main:app  # 让 /chat 的工具调用经 MCP stdio
```

## 真实向量库与缓存（第5周）
- `app/rag/qdrant_store.py`：Qdrant 适配器，与内存版 `VectorStore` 同接口（add/search/chunks…），可直接替换。
- `app/rag/store_factory.py`：按 `VECTOR_BACKEND` 选 `memory`/`qdrant`；**Qdrant 不可用时自动回退内存**，任何环境可跑。
- `app/rag/cache.py`：embedding 缓存，优先 Redis（`REDIS_URL`），不可用时进程内 dict 回退，降低重复检索的计算/API 开销。
- `docker-compose.yml`：一键起 `api + qdrant + redis`，api 启动前自动建索引。

一键起完整栈：
```bash
docker compose up --build       # qdrant + redis + api，全部接好
# 本地开发默认内存库 + 内存缓存，零外部依赖也能跑
```

## 当前进度（对照技术设计文档里程碑）
- [x] 第1周：朴素 RAG + FastAPI + LangGraph 三节点编排
- [x] 第2周：查询改写(multi-query) + 混合检索(向量+BM25, RRF) + Rerank + eval(hit@k/MRR)
- [x] 第3周：tool 节点 + Critic 反思节点 + faithfulness 重试循环（max_iterations 限制）
- [x] 第4周(部分)：MCP 风格工具层（工具名/参数校验、输出截断、错误处理）、安全计算器(AST 白名单)、prompt injection 分离
- [x] 第4周(余)：真正的 MCP Server —— JSON-RPC 2.0 over stdio（initialize/tools/list/tools/call）+ 客户端握手，USE_MCP=1 切换
- [x] 第5周(部分)：Qdrant 向量库适配器 + Redis embedding 缓存 + docker-compose(api+qdrant+redis)，均带回退
- [x] 第5周(余)：LLM-judge faithfulness（有 key 走事实核查打分，无则回退启发式）+ 生成侧评测脚本
- [x] 第6周：架构图(docs/architecture.svg) + Demo(scripts/demo.py) + 简历/面试要点定稿

## 扩展点（代码里已留 TODO）
- `graph/nodes.py::planner_node` → 加查询改写。
- `rag/retriever.py` → 加混合检索 + Rerank。
- `graph/build_graph.py` → 加 tool/critic 节点与条件边。
- `rag/store.py` → 替换为 Qdrant/PGVector（保持 add/search 接口）。
