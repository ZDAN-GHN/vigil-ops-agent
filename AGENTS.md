# AGENTS.md

## 语言规范

- 对话、注释、文档使用**简体中文**；代码标识符使用**英文**。
- Commit Message 使用简体中文。

## 技术栈

| 项 | 选择 |
|----|------|
| Python | `>=3.11,<3.14`（当前 3.13） |
| 包管理 | **uv**（非 pip/poetry） |
| Web 框架 | FastAPI + uvicorn |
| Agent 框架 | LangGraph + langchain-qwq（ChatQwen 原生集成） |
| 向量库 | Milvus（`langchain-milvus` + `pymilvus`） |
| MCP | `langchain-mcp-adapters` + `fastmcp` |
| 日志 | **Loguru**（`from loguru import logger`，非标准 logging） |
| 数据库 | MySQL（`aiomysql` + `SQLAlchemy`）+ Redis（checkpoint + 消息队列） |

## 常用命令

```bash
# 环境初始化
uv pip install -e .            # 安装依赖
uv pip install -e ".[dev]"     # 安装开发依赖
make init                      # 一键初始化（Docker Milvus + 服务 + 上传文档）

# 服务管理
make start                     # 启动全部（MCP + FastAPI）
make stop                      # 停止全部
make dev                       # 开发模式（前台热重载）
.\start-windows.bat            # Windows 启动
.\stop-windows.bat             # Windows 停止

# 代码质量（PR 前必须通过）
make format                    # ruff format + isort
make lint                      # ruff check
make test                      # pytest --cov
make check-all                 # format + lint + test

# 文档向量化
make upload                    # 上传 aiops-docs/*.md 到 Milvus
```

## 服务与端口

| 服务 | 端口 | 入口 |
|------|------|------|
| FastAPI 主服务 | `9999` | `uvicorn app.main:app` |
| CLS MCP 服务 | `8003` | `mcp_servers/cls_server.py` |
| Monitor MCP 服务 | `8004` | `mcp_servers/monitor_server.py` |
| Milvus 向量库 | `19530` | Docker `milvus-standalone` |

## 架构要点

分层：`api/`（路由薄层）→ `services/`（业务逻辑）→ `agent/`（LangGraph 节点 + MCP 客户端）→ `tools/`（工具）→ `models/`（Pydantic/ORM）→ `core/`（基础设施：LLM 工厂、DB 连接）→ `utils/`。

关键模式：
- **全局服务单例**：`rag_agent_service`、`vector_store_manager` 等模块级实例，直接 `import` 使用。
- **MCP 聚合**：`agent/mcp_client.py` 用 `MultiServerMCPClient` 连接多个 MCP 服务器，`retry_interceptor` 提供指数退避重试。
- **SSE 流式**：`/api/chat/stream` 和 `/api/aiops` 通过 `sse-starlette` 返回 JSON 事件。
- **三级消息兜底**：Redis List → 内存队列 → 兜底日志文件（`logs/mq_fallback.jsonl`）。
- **Redis checkpoint + MySQL 备份**：短期记忆 Redis TTL=7天，过期自动从 MySQL 恢复。

两条核心路径：
- **RAG 对话**：`chat.py` → `rag_agent_service` → LangGraph Agent → 工具调用（Milvus 检索 / MCP）→ SSE 流式 → 异步持久化 MySQL
- **AIOps 诊断**：`aiops.py` → `aiops_service` → Plan-Execute-Replan 循环 → SSE 流式

## 禁止事项

- **不要**直接修改 `core/redis_checkpointer.py`、`core/milvus_client.py`、`core/mysql_client.py`，除非明确要求。
- **不要**修改 MCP 服务器接口（`mcp_servers/`）而不通知。
- **不要**使用 `pip install`，始终使用 `uv pip install`。
- **不要**使用标准 `logging`，始终使用 `from loguru import logger`。
- **不要**硬编码环境变量，所有配置通过 `config.py`（Pydantic Settings）从 `.env` 加载。
- **不要**在 `app/` 下创建新的顶层包，除非讨论过。

## 易踩坑

- `DASHSCOPE_API_BASE` 必须为 `https://dashscope.aliyuncs.com/compatible-mode/v1`，否则默认访问新加坡站点导致鉴权失败。
- `langchain-qwq` 是 ChatQwen 原生集成，非通用 OpenAI 兼容适配器，不要混用 `ChatOpenAI`。
- 虚拟环境在 `.venv/`，由 uv 管理，不要手动创建或激活其他虚拟环境。
- `RERANK_ENABLED` 默认 `False`，启用 Rerank 需要同时在 `.env` 中配置。
- 认证模块：`core/auth.py`（JWT 令牌处理）+ `services/auth_service.py`（业务逻辑）+ `api/auth.py`（路由），修改时需三层联动。

## 模块约束

- 单模块目标 **< 500 行**（不含测试）。超过约 800 行应拆分。
- `services/` 中新增服务时，保持单一职责，一个文件对应一个领域。
- 新增 Agent 工具放在 `tools/`，遵循 `knowledge_tool.py` 的签名模式。

## 完成标准

修改代码后，满足以下条件才算完成：

1. `make format && make lint` 无错误
2. `make test` 全部通过（新增功能需补充测试）
3. 新增 API 端点必须在 `models/` 中有对应的请求/响应模型
4. 新增环境变量必须在 `config.py` 中声明并设置合理默认值
5. 日志使用 `logger`（Loguru），不使用 `print`

## 验证命令

| 变更类型 | 验证命令 |
|----------|----------|
| 业务逻辑 / API | `make check-all` |
| 向量检索 / RAG | `make test` + 手动 `curl` 测试 `/api/chat_stream` |
| MCP 工具 | `make status-mcp` 确认服务在线 |
| 配置变更 | `make start` + `curl localhost:9999/health` |
| 认证模块 | `make test` + 检查 JWT 令牌流程 |
