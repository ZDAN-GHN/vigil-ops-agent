# AGENTS.md

该文件为在本仓库中进行代码工作的所有python开发者提供了指导

## 语言规范

- 所有对话、解释、注释、文档必须使用**简体中文**。
- 代码中的变量名、函数名、类名等标识符必须使用**英文**，遵循通用命名规范。
- Commit Message 必须使用简体中文。

## 常用命令

### 环境

- Python 版本：`3.13`（见 `.python-version`），`pyproject.toml` 要求 `>=3.11,<3.14`
- 包管理器：**uv**（非 pip/poetry）
- 虚拟环境：`.venv/`（uv 管理）
- 环境变量：`.env` 文件（DashScope API Key、Milvus 连接等）

### 安装与启动

```bash
# 安装依赖
uv pip install -e .
# 或安装开发依赖（pytest, ruff, black, mypy 等）
uv pip install -e ".[dev]"

# 一键初始化（Docker Milvus + 服务 + 上传文档）
make init

# 启动所有服务（MCP + FastAPI）
make start

# 停止所有服务
make stop

# Windows 用户（无 make）
.\start-windows.bat   # 启动
.\stop-windows.bat    # 停止
```

### 服务组成

启动后共有 **4 个进程**：

| 服务             | 端口      | 说明                                            |
| ---------------- | --------- | ----------------------------------------------- |
| FastAPI 主服务   | `9999`  | `uvicorn app.main:app`，Web UI + API          |
| CLS MCP 服务     | `8003`  | `mcp_servers/cls_server.py`，日志查询工具     |
| Monitor MCP 服务 | `8004`  | `mcp_servers/monitor_server.py`，监控数据工具 |
| Milvus 向量库    | `19530` | Docker 容器`milvus-standalone`                |

### 开发模式

```bash
# 前台热重载开发
make dev

# 单独管理各服务
make start-cls       # 启动 CLS MCP
make start-monitor   # 启动 Monitor MCP
make start-api       # 启动 FastAPI
make status-mcp      # 查看 MCP 服务状态
```

### 代码质量

```bash
make format          # ruff format + isort（line-length=100）
make lint            # ruff check
make fix             # ruff check --fix + format
make type-check      # mypy app/ --ignore-missing-imports
make test            # pytest tests/ --cov=app
make test-quick      # pytest tests/ -v（无覆盖率）
make check-all       # format + lint + test
```

### 文档管理

```bash
make upload          # 上传 aiops-docs/*.md 到向量库
make list-docs       # 列出可上传文档
```

### 预提交

```bash
make pre-commit-install   # 安装 hooks（isort + black + ruff + bandit + docformatter + commitizen）
make pre-commit           # 手动运行所有 hooks
```

## 架构总览

```
app/
├── main.py            # FastAPI 入口，lifespan 管理 Milvus 连接
├── config.py          # Pydantic Settings，从 .env 加载配置
├── api/               # 路由层 —— 薄层，只做请求/响应转换
│   ├── chat.py        # /api/chat, /api/chat_stream (SSE), /api/chat/clear
│   ├── aiops.py       # /api/aiops (SSE 流式诊断)
│   ├── file.py        # /api/upload (文档上传 → 向量化)
│   └── health.py      # /health
├── services/          # 业务逻辑层
│   ├── rag_agent_service.py   # RAG Agent —— LangGraph ReAct 对话代理
│   ├── aiops_service.py       # AIOps —— LangGraph Plan-Execute-Replan 工作流
│   ├── vector_store_manager.py    # Milvus VectorStore 封装
│   ├── vector_embedding_service.py # Embedding 服务（DashScope text-embedding-v4）
│   ├── vector_index_service.py    # 向量索引（写入 Milvus）
│   ├── vector_search_service.py   # 向量检索
│   └── document_splitter_service.py # 文档分块（RecursiveCharacterTextSplitter）
├── agent/             # Agent 核心
│   ├── mcp_client.py  # MultiServerMCPClient 全局单例 + 重试拦截器
│   └── aiops/         # Plan-Execute-Replan 三节点
│       ├── state.py       # PlanExecuteState (TypedDict)
│       ├── planner.py     # 制定执行计划
│       ├── executor.py    # 执行步骤（调用工具）
│       └── replanner.py   # 评估结果 → continue / replan / respond
├── tools/             # Agent 工具集
│   ├── knowledge_tool.py  # 知识检索（从 Milvus 检索 → 返回上下文）
│   └── time_tool.py       # 获取当前时间
├── models/            # Pydantic 请求/响应模型
├── core/              # 基础设施
│   ├── llm_factory.py     # LLM 工厂（DashScope/ChatQwen 实例化）
│   └── milvus_client.py   # Milvus 连接管理（单例）
└── utils/
    └── logger.py      # Loguru 日志配置（控制台 + 按天轮转文件）
```

### 两条核心数据流

**1. RAG 对话（`/api/chat_stream`）**

```
用户请求 → chat.py → rag_agent_service.query_stream()
  → LangGraph Agent（ChatQwen + MemorySaver）
    → trim_messages_middleware 修剪历史（保留系统消息 + 最近 6 条）
    → Agent 决定是否调用工具：
       - retrieve_knowledge → vector_store_manager → Milvus 检索
       - MCP 工具（cls/monitor）→ mcp_client
    → 流式输出 SSE 事件：tool_call / content / done / error
```

**2. AIOps 诊断（`/api/aiops`）**

```
用户请求 → aiops.py → aiops_service.run()
  → LangGraph StateGraph(PlanExecuteState)
    → Planner：分析任务，生成步骤列表（Plan pydantic model）
    → Executor：取 plan[0]，用 ToolNode 执行（本地工具 + MCP 工具）
    → Replanner：评估结果 → Action(continue/replan/respond)
    → 条件边：有 response → END，有剩余 plan → Executor，否则 → END
  → 流式输出诊断过程 + 最终报告
```

### 关键设计模式

- **LangGraph StateGraph + MemorySaver**：对话和 AIOps 都使用 checkpointer 实现会话记忆，通过 `thread_id` 区分会话。
- **全局服务单例**：`rag_agent_service`、`vector_store_manager`、`milvus_manager` 等模块级实例，在 `services/` 和 `core/` 中直接导入使用。
- **MCP 工具聚合**：`mcp_client.py` 用 `MultiServerMCPClient` 连接多个 MCP 服务器，`retry_interceptor` 提供指数退避重试。
- **SSE 流式协议**：`/api/chat_stream` 和 `/api/aiops` 都通过 `sse-starlette` 返回，data 字段为 JSON（`type` 区分事件类型）。
- **向量化管道**：文件上传 → `document_splitter_service` 分块 → `vector_embedding_service` embedding → `vector_index_service` 写入 Milvus collection `biz`。

### 配置文件

- `config.py`：所有配置通过 Pydantic Settings 从 `.env` 加载，全局 `config` 单例
- 关键环境变量：`DASHSCOPE_API_KEY`、`DASHSCOPE_API_BASE`、`MILVUS_HOST/PORT`、`RAG_TOP_K`、`CHUNK_MAX_SIZE/OVERLAP`
- MCP 服务器地址在 config 中配置（默认 `localhost:8003` 和 `localhost:8004`）

### 日志

- 使用 **Loguru**（非标准 logging），全局 `from loguru import logger`
- 控制台 + 文件（`logs/app_YYYY-MM-DD.log`，按天轮转，保留 7 天，zip 压缩）
- Debug 模式（`config.debug=True`）下显示完整异常栈和变量值

### 依赖管理

- 依赖定义在 `pyproject.toml`，锁定文件为 `uv.lock`
- LLM：`langchain-qwq`（ChatQwen 原生集成，非通用 OpenAI 兼容）
- 向量库：`langchain-milvus` + `pymilvus`
- MCP：`langchain-mcp-adapters` + `fastmcp`
- 注意 `DASHSCOPE_API_BASE` 必须配置为 `https://dashscope.aliyuncs.com/compatible-mode/v1`，否则默认访问新加坡站点
