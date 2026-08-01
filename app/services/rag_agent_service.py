"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from textwrap import dedent
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage, HumanMessageChunk, AIMessage, AIMessageChunk,
)
from langchain_qwq import ChatQwen
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from loguru import logger
from typing_extensions import TypedDict

from app.agent.context import AgentContext
from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.core.feature_extraction_middleware import FeatureExtractionMiddleware
from app.core.redis_checkpointer import AsyncRedisSaver
from app.services.long_term_memory_service import long_term_memory_service
from app.tools import get_current_time, retrieve_knowledge


# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key
class AgentState(TypedDict):
    """Agent 状态"""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class _SummarySetting(TypedDict):
    """对话摘要设置"""

    summary_enabled: bool
    summary_model: str | BaseChatModel | None
    summary_trigger_messages: int
    summary_trigger_tokens: int
    summary_trigger_fraction: float
    summary_keep_messages: int
    summary_prompt: str | None


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(
            self,
            system_prompt: str,
            model: str | BaseChatModel = config.rag_model,
            streaming: bool = True,
            temperature: float = 1.0,
            summary_setting=None,
    ):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
            summary_setting: 对话摘要配置
        """
        # 设置系统提示词
        self.system_prompt = system_prompt
        self.streaming = streaming
        self.temperature = temperature
        # 对话摘要配置
        # 绑定模型
        self._bind_model(model)
        self._bind_summary_setting(summary_setting)

        # 定义基础工具
        self.tools = [retrieve_knowledge, get_current_time]

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 创建 Redis 检查点（热数据，用于会话持久化）
        self.checkpointer: AsyncRedisSaver | None = None

        # PostgreSQL 冷 checkpointer（Redis TTL 过期后 fallback，由 main.py lifespan 注入）
        self.postgres_saver: AsyncPostgresSaver | None = None

        # LangGraph Store（长期记忆，由 main.py lifespan 注入）
        self.store: BaseStore = None

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(
            f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model.model}, "
            f"streaming={streaming}, summary_enabled={self.summary_enabled}"
        )

    # 绑定模型
    def _bind_model(self, model: str | BaseChatModel | None):
        if isinstance(model, str) or model is None:
            self.model = init_chat_model(
                model=model if isinstance(model, str) else config.rag_model,
                api_key=config.dashscope_api_key,
                temperature=self.temperature,
                streaming=self.streaming,
            )
        else:
            self.model = model

    # 将对话摘要设置绑定到对象
    def _bind_summary_setting(self, summary_setting: _SummarySetting):
        # 保证摘要设置非空
        summary_setting = {} if not summary_setting else summary_setting

        # 是否开启摘要
        self.summary_enabled = (
            True
            if not summary_setting.get("summary_enabled", None)
            else summary_setting["summary_enabled"]
        )

        # 如果不开启摘要,直接跳过
        if not self.summary_enabled:
            return

        # 摘要模型
        self.summary_model = (
            self.model
            if not summary_setting.get("summary_model", None)
            else summary_setting["summary_model"]
        )

        # 摘要触发消息阈值
        self.summary_trigger_messages = (
            25
            if not summary_setting.get("summary_trigger_messages", None)
            else summary_setting["summary_trigger_messages"]
        )

        # 摘要触发 tokens 阈值
        self.summary_trigger_tokens = (
            25
            if not summary_setting.get("summary_trigger_tokens", None)
            else summary_setting["summary_trigger_tokens"]
        )

        # 摘要触发 fraction 阈值（上下文长度比例。历史token的累计数量达到模型的 max_input_tokens*fraction 触发摘要）
        self.summary_trigger_fraction = (
            0.8
            if not summary_setting.get("summary_trigger_fraction", None)
            else summary_setting["summary_trigger_fraction"]
        )

        # 触发摘要时，保留最新消息条数
        self.summary_keep_messages = (
            10
            if not summary_setting.get("summary_keep_messages", None)
            else summary_setting["summary_keep_messages"]
        )

        _DEFAULT_SUMMARY_PROMPT = dedent("""
            你是一个对话上下文提取助手。请从以下对话历史中提取最关键的信息，生成一份简洁的摘要。

            这份摘要将替换原始对话历史，成为后续对话的上下文。
            请确保摘要包含以下结构化内容：

            ## 会话意图
            用户的核心目标或请求是什么？

            ## 关键上下文
            提取对话中最重要的信息，包括重要的结论、选择和决策依据。

            ## 已执行操作
            对话中调用了哪些工具？得到了什么结果？做出了哪些关键判断？

            ## 待办事项
            还有哪些未完成的任务？接下来应该做什么？

            请只输出提取的摘要内容，不要添加额外说明。

            {messages}
            """).strip()

        # 摘要提示词
        self.summary_prompt = (
            _DEFAULT_SUMMARY_PROMPT
            if not summary_setting.get("summary_prompt", None)
            else summary_setting["summary_prompt"]
        )

    # region 私有方法

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具和摘要中间件）"""
        if self._agent_initialized:
            return

        # 使用全局 MCP 客户端管理器（带重试拦截器）
        # MCP 连接失败时优雅降级：仅使用本地工具，不阻塞 Agent 启动
        try:
            mcp_client = await get_mcp_client_with_retry()
            mcp_tools = await mcp_client.get_tools()
            logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")
            self.mcp_tools = mcp_tools
        except Exception as e:
            self._log_exception_group("MCP 工具加载失败，将仅使用本地工具", e)
            self.mcp_tools = []

        # 合并所有工具
        all_tools = self.tools + self.mcp_tools

        # 懒加载 Redis CheckPointer
        if self.checkpointer is None:
            from app.core.manager.redis_client import redis_manager

            redis_client = await redis_manager.get_client()
            self.checkpointer = AsyncRedisSaver(redis_client, ttl=config.redis_checkpoint_ttl)
            logger.info("Redis CheckPointer 初始化完成")

        # 构建中间件列表
        middleware = []

        # 用户画像特征提取中间件：从 Store 加载/保存用户画像
        if self.store is not None and config.long_term_memory_enabled:
            feature_middleware = FeatureExtractionMiddleware(
                store=self.store,
                llm=self.model,
            )
            middleware.append(feature_middleware)
            logger.info("用户画像特征提取中间件已启用（长期记忆）")

        # 对话摘要中间件：当消息数超过阈值时，自动用 LLM 生成摘要替换旧消息
        if self.summary_enabled:
            # 中文对话摘要提示词 —— 用于 SummarizationMiddleware，当消息历史超过阈值时自动摘要
            # 注意：{messages} 占位符是 SummarizationMiddleware 的硬性要求，不可删除或改名
            summary_middleware = SummarizationMiddleware(
                model=self.summary_model,
                trigger=[
                    ("messages", self.summary_trigger_messages),
                    ("tokens", self.summary_trigger_tokens),
                    ("fraction", self.summary_trigger_fraction),
                ],
                keep=("messages", self.summary_keep_messages),
                summary_prompt=self.summary_prompt,
            )
            middleware.append(summary_middleware)
            logger.info(
                f"对话摘要中间件已启用: trigger={self.summary_trigger_messages} 条消息, "
                f"keep={self.summary_keep_messages} 条消息"
            )

        self.agent = create_agent(
            model=self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
            store=self.store,
            middleware=middleware,
            context_schema=AgentContext,
        )

        self._agent_initialized = True

        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    async def _ensure_checkpoint_restored(self, session_id: str) -> None:
        """确保 Redis checkpoint 存在，若不存在则从 PostgreSQL 冷存储恢复

        当 Redis TTL 过期导致 checkpoint 丢失时，从 PostgreSQL 冷 checkpoint
        加载历史并写回 Redis checkpointer，使 LangGraph 能恢复上下文。

        Args:
            session_id: 会话 ID
        """
        if not config.conversation_history_enabled:
            return

        if self.checkpointer is None:
            return

        # 检查 Redis 中是否已有 checkpoint
        cfg = {"configurable": {"thread_id": session_id}}
        checkpoint_tuple = await self.checkpointer.aget_tuple(cfg)
        if checkpoint_tuple is not None:
            # Redis 中有数据，无需恢复
            return

        # Redis 中无数据，尝试从 PostgreSQL 冷 checkpoint 恢复
        if self.postgres_saver is None:
            logger.info(
                f"[会话 {session_id}] Redis checkpoint 不存在，且 PostgreSQL 冷存储未初始化"
            )
            return

        logger.info(f"[会话 {session_id}] Redis checkpoint 不存在，尝试从 PostgreSQL 恢复...")
        pg_config = {"configurable": {"thread_id": session_id}}
        pg_tuple = await self.postgres_saver.aget_tuple(pg_config)
        if pg_tuple is None:
            logger.info(f"[会话 {session_id}] PostgreSQL 中也无 checkpoint")
            return

        # 将 PostgreSQL 中的 checkpoint 恢复到 Redis
        restore_config = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": "",
            }
        }
        await self.checkpointer.aput(
            restore_config,
            pg_tuple.checkpoint,
            pg_tuple.metadata or {},
            pg_tuple.checkpoint.get("channel_versions", {}),
        )

        # 设置 TTL
        if self.checkpointer.ttl > 0:
            key = self.checkpointer._key(session_id, "")
            await self.checkpointer.conn.expire(key, config.conversation_history_redis_ttl)

        logger.info(
            f"[会话 {session_id}] 已从 PostgreSQL 恢复 checkpoint 到 Redis "
            f"(TTL={config.conversation_history_redis_ttl}s)"
        )

    async def _persist_to_postgres(self, session_id: str, async_: bool = True) -> None:
        """将 Redis checkpoint 持久化到 PostgreSQL 冷存储

        Agent 执行完成后，将 Redis 中的最新 checkpoint 写入 PostgreSQL，
        确保 Redis TTL 过期后仍可从冷存储恢复对话上下文。

        Args:
            session_id: 会话 ID
            async_: 是否异步执行（后台任务，不阻塞主流程），默认为 True
        """
        if not config.conversation_history_enabled:
            return

        if self.checkpointer is None or self.postgres_saver is None:
            return

        if async_:
            asyncio.create_task(self._do_persist_to_postgres(session_id))
        else:
            await self._do_persist_to_postgres(session_id)

    async def _do_persist_to_postgres(self, session_id: str) -> None:
        """实际执行 checkpoint 持久化到 PostgreSQL 的内部方法

        Args:
            session_id: 会话 ID
        """
        try:
            cfg = {"configurable": {"thread_id": session_id}}
            checkpoint_tuple = await self.checkpointer.aget_tuple(cfg)
            if checkpoint_tuple is None:
                return

            restore_config = {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": "",
                }
            }
            await self.postgres_saver.aput(
                restore_config,
                checkpoint_tuple.checkpoint,
                checkpoint_tuple.metadata or {},
                checkpoint_tuple.checkpoint.get("channel_versions", {}),
            )
            logger.info(f"[会话 {session_id}] checkpoint 已持久化到 PostgreSQL")
        except Exception as e:
            logger.warning(f"[会话 {session_id}] 持久化 checkpoint 到 PostgreSQL 失败: {e}")

    async def _build_system_prompt_with_profile(
            self,
            user_id: str | None,
    ) -> str:
        """构建包含用户画像的系统提示词

        加载用户画像并格式化为文本，拼接到系统提示词开头，
        确保模型优先看到用户信息（而非追加到消息列表末尾）。

        Args:
            user_id: 用户 ID（可选）

        Returns:
            拼接后的系统提示词字符串
        """
        if not user_id or not self.store or not config.long_term_memory_enabled:
            return self.system_prompt

        try:
            profile = await long_term_memory_service.load_user_profile(self.store, user_id)
            if not profile:
                return self.system_prompt

            profile_text = long_term_memory_service.format_profile_for_prompt(profile)
            if not profile_text:
                return self.system_prompt

            logger.info(f"[用户 {user_id}] 注入用户画像到系统提示词结尾")
            return f"{self.system_prompt}\n\n{profile_text}"

        except Exception as e:
            logger.warning(f"[用户 {user_id}] 加载用户画像失败，使用默认系统提示词: {e}")
            return self.system_prompt

    # endregion

    # region 对外服务方法

    async def query(
            self,
            question: str,
            session_id: str,
            user_id: str | None = None,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）
            user_id: 用户 ID（可选，用于长期记忆）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            # 确保 Redis checkpoint 存在（若过期则从 MySQL 恢复）
            await self._ensure_checkpoint_restored(session_id)

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            # 构建消息列表（系统提示 + 用户问题）
            # 画像注入到 system prompt 结尾，确保模型优先看到用户信息
            system_content = await self._build_system_prompt_with_profile(user_id)
            messages = [SystemMessage(content=system_content), HumanMessage(content=question)]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（会话持久化）
            config_dict = {"configurable": {"thread_id": session_id}}

            # context 作为独立关键字参数传递（长期记忆，供 Middleware 读取）
            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
                context=AgentContext(user_id=str(user_id) if user_id else None),
            )

            # 异步持久化到 PostgreSQL 冷存储（不阻塞主流程）
            await self._persist_to_postgres(session_id, async_=True)

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                # 从后往前查找第一个非摘要消息
                answer = ""
                for msg in reversed(messages_result):

                    # 跳过摘要消息（由 SummarizationMiddleware 注入，不应显示在前端）
                    if (
                            isinstance(msg, HumanMessage)
                            and getattr(msg, "additional_kwargs", {}).get("lc_source") == "summarization"
                    ):
                        logger.info(f"[会话 {session_id}] 跳过摘要消息")
                        continue

                    answer = msg.content if hasattr(msg, "content") else str(msg)

                    # 记录工具调用
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_names = [tc.get("name", "unknown") for tc in msg.tool_calls]
                        logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")
                    break

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            self._log_exception_group(f"[会话 {session_id}] RAG Agent 查询失败（非流式）", e)
            raise

    async def query_stream(
            self,
            question: str,
            session_id: str,
            user_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）
            user_id: 用户 ID（可选，用于长期记忆）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()

            # 确保 Redis checkpoint 存在（若过期则从 MySQL 恢复）
            await self._ensure_checkpoint_restored(session_id)

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            # 构建消息列表（系统提示 + 用户问题）
            # 画像注入到 system prompt 开头，确保模型优先看到用户信息
            system_content = await self._build_system_prompt_with_profile(user_id)
            messages = [SystemMessage(content=system_content), HumanMessage(content=question)]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（会话持久化）
            config_dict = {"configurable": {"thread_id": session_id}}

            # context 作为独立关键字参数传递（长期记忆，供 Middleware 读取）
            async for token, metadata in self.agent.astream(
                    input=agent_input,
                    config=config_dict,
                    context=AgentContext(user_id=str(user_id) if user_id else None),
                    stream_mode="messages",
            ):
                # 元数据，筛选消息、日志输出
                node_name = ""
                if isinstance(metadata, dict):
                    # 检查是否是摘要中间件的输出（通过 metadata 中的节点名称判断）
                    # 获取节点名
                    node_name = (metadata.get("langgraph_node") or "")
                    logger.info(f"接收来自{node_name}节点的消息，元数据:{metadata}")
                    # 摘要中间件的节点名形如：SummarizationMiddleware.before_model
                    if node_name.startswith("SummarizationMiddleware"):
                        logger.info(f"[会话 {session_id}] 跳过摘要中间件输出（节点: {node_name}）")
                        continue

                # logger.info(f"流式消息:{token},类型:{type(token)},metadata:{metadata}")

                # 跳过摘要消息（由 SummarizationMiddleware 注入，不应显示在前端）
                # 检查 HumanMessage 类型的摘要（中间件直接注入的）
                if (
                        isinstance(token, HumanMessage | HumanMessageChunk)
                        and getattr(token, "additional_kwargs", {}).get("lc_source") == "summarization"
                ):
                    logger.info(f"[会话 {session_id}] 跳过摘要消息（HumanMessage）")
                    continue

                # 跳过 RemoveMessage（摘要中间件清理消息时使用）
                if type(token).__name__ == "RemoveMessage":
                    logger.info(f"[会话 {session_id}] 跳过 RemoveMessage")
                    continue

                # 跳过模型思考消息
                if (
                        isinstance(token, AIMessage | AIMessageChunk)
                        and "reasoning_content" in getattr(token, "additional_kwargs", {})
                ):
                    logger.debug(f"[会话 {session_id}] 跳过思考过程")
                    continue

                # content_blocks 中包含了多种数据: text - 文本; image-图片 ... HumanMessage 也可以包含这一属性
                content_blocks = getattr(token, "content_blocks", None)

                # 从 content_blocks 中获取数据，并转换为前端期望格式
                if content_blocks and isinstance(content_blocks, list):
                    for block in content_blocks:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content = block.get("text", "")
                            if text_content:
                                yield {
                                    "type": "content",
                                    "data": text_content,
                                    "node": node_name,
                                }

            # 流式完成后，异步持久化到 PostgreSQL 冷存储（不阻塞主流程）
            await self._persist_to_postgres(session_id, async_=True)

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            self._log_exception_group(f"[会话 {session_id}] RAG Agent 查询失败（流式）", e)
            yield {"type": "error", "data": str(e)}
            raise

    async def get_chat_history(self, session_id: str) -> list:
        """
        获取会话历史（优先从 Redis checkpointer 读取，Redis 无数据时从 PostgreSQL 加载）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant|summary", "content": "...", "timestamp": "..."}]
        """
        try:
            messages: list[BaseMessage] = []
            source = "none"

            # 1. 优先从 Redis checkpointer 读取
            if self.checkpointer is not None:
                cfg = {"configurable": {"thread_id": session_id}}
                checkpoint_tuple = await self.checkpointer.aget_tuple(cfg)
                if checkpoint_tuple:
                    checkpoint_data = checkpoint_tuple.checkpoint
                    messages = checkpoint_data.get("channel_values", {}).get("messages", [])
                    source = "redis"

            # 2. Redis 无数据时，从 PostgreSQL 冷 checkpoint 加载
            if (
                    not messages
                    and config.conversation_history_enabled
                    and self.postgres_saver is not None
            ):
                pg_config = {"configurable": {"thread_id": session_id}}
                pg_tuple = await self.postgres_saver.aget_tuple(pg_config)
                if pg_tuple:
                    messages = pg_tuple.checkpoint.get("channel_values", {}).get("messages", [])
                    if messages:
                        source = "postgres"
                        logger.info(
                            f"[会话 {session_id}] Redis 无数据，从 PostgreSQL 加载了 {len(messages)} 条历史"
                        )

            if not messages:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            # 转换为前端需要的格式
            history = []
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg, SystemMessage):
                    continue

                # 跳过摘要消息（由 SummarizationMiddleware 注入，不应显示在前端）
                if (
                        isinstance(msg, HumanMessage)
                        and getattr(msg, "additional_kwargs", {}).get("lc_source") == "summarization"
                ):
                    logger.info(f"[会话 {session_id}] 跳过摘要消息")
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, "content") else str(msg)

                # 提取时间戳（如果有的话）
                timestamp = getattr(msg, "timestamp", None)
                if timestamp:
                    history.append({"role": role, "content": content, "timestamp": timestamp})
                else:
                    from datetime import datetime

                    history.append(
                        {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
                    )

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}, 来源: {source}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    async def clear_chat_history(self, session_id: str) -> bool:
        """
        清空会话历史（同时删除 Redis checkpoint、PostgreSQL 冷 checkpoint 和 PostgreSQL 对话历史）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        try:
            # 1. 删除 Redis checkpoint
            if self.checkpointer is not None:
                await self.checkpointer.adelete_thread(session_id)
                logger.info(f"已清除 Redis checkpoint: {session_id}")

            # 2. 删除 PostgreSQL 冷 checkpoint
            if self.postgres_saver is not None:
                try:
                    await self.postgres_saver.adelete_thread(session_id)
                    logger.info(f"已清除 PostgreSQL 冷 checkpoint: {session_id}")
                except Exception as pg_err:
                    logger.warning(f"清除 PostgreSQL checkpoint 失败（非致命）: {pg_err}")

            logger.info(f"已清除会话历史: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")

    # endregion

    @staticmethod
    def _log_exception_group(prefix: str, exc: BaseException) -> None:
        """解包 ExceptionGroup / BaseExceptionGroup，逐条记录子异常详情

        Python 3.11+ 的 asyncio.TaskGroup 和 anyio TaskGroup 会将多个子异常
        包装为 ExceptionGroup。直接 str(e) 只会显示外壳信息（如
        'unhandled errors in a TaskGroup (1 sub-exception)'），子异常被吞掉。
        此方法递归解包，确保每条子异常都被独立记录。

        Args:
            prefix: 日志前缀（如 '[会话 xxx] RAG Agent 查询失败'）
            exc: 捕获到的异常
        """
        sub_exceptions = getattr(exc, "exceptions", None)
        if sub_exceptions:
            logger.error(f"{prefix}: {type(exc).__name__}（{len(sub_exceptions)} 个子异常）")
            for i, sub in enumerate(sub_exceptions):
                logger.error(f"  └─ 子异常[{i}] {type(sub).__name__}: {sub}")
        else:
            logger.error(f"{prefix}: {type(exc).__name__}: {exc}")


# 全局单例 - 启用流式输出，摘要中间件默认开启
# 可通过参数自定义摘要行为：
#   summary_enabled=False          关闭摘要
#   summary_trigger_messages=10    触发摘要的消息数阈值
#   summary_keep_messages=6        摘要后保留最近消息数
#   summary_model="qwen-turbo"     使用独立模型做摘要
DEFAULT_RAG_PROMPT = dedent("""
            你是一个专业的 AIOps 智能运维助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            用户画像:
            对话中可能会在系统提示开头注入【用户画像】信息，其中包含当前用户的关键事实和偏好。
            当用户询问个人信息（如姓名、偏好、历史话题等）时，请优先从用户画像中查找答案。

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()
rag_agent_service = RagAgentService(
    model=ChatQwen(
        name=config.rag_model, temperature=1.0, profile={"max_input_tokens": 500 * 1024}
    ),
    system_prompt=DEFAULT_RAG_PROMPT,
    summary_setting={
        "summary_enabled": True,
        "summary_trigger_messages": 6,
        "summary_trigger_tokens": 400,
        "summary_trigger_fraction": 0.1,
        "summary_keep_messages": 4,
    },
)
