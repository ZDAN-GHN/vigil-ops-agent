"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

from typing import Annotated, Any, AsyncGenerator, Dict, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict
from langchain_qwq import ChatQwen

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from app.core.redis_checkpointer import AsyncRedisSaver
from app.services.conversation_history_service import conversation_history_service

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key

# 中文对话摘要提示词 —— 用于 SummarizationMiddleware，当消息历史超过阈值时自动摘要
# 注意：{messages} 占位符是 SummarizationMiddleware 的硬性要求，不可删除或改名
SUMMARY_PROMPT_ZH = """你是一个对话上下文提取助手。请从以下对话历史中提取最关键的信息，生成一份简洁的摘要。

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

<messages>
{messages}
</messages>"""


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(
        self,
        streaming: bool = True,
        summary_enabled: bool = True,
        summary_trigger_messages: int = 10,
        summary_keep_messages: int = 6,
        summary_model: str | None = None,
    ):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
            summary_enabled: 是否启用对话摘要中间件，默认为 True
            summary_trigger_messages: 触发摘要的消息数阈值，默认为 10
            summary_keep_messages: 摘要后保留最近的消息数，默认为 6
            summary_model: 摘要使用的模型名称，为 None 时复用主模型，默认为 None
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()

        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
        )

        # 对话摘要配置
        self.summary_enabled = summary_enabled
        self.summary_trigger_messages = summary_trigger_messages
        self.summary_keep_messages = summary_keep_messages
        self.summary_model = summary_model

        # 定义基础工具
        self.tools = [retrieve_knowledge, get_current_time]

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 创建 Redis 检查点（用于会话持久化，支持重启后恢复）
        self.checkpointer: AsyncRedisSaver | None = None

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(
            f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, "
            f"streaming={streaming}, summary_enabled={summary_enabled}"
        )

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

        # 懒加载 Redis Checkpointer
        if self.checkpointer is None:
            from app.core.redis_client import redis_manager

            redis_client = await redis_manager.get_client()
            self.checkpointer = AsyncRedisSaver(
                redis_client, ttl=config.redis_checkpoint_ttl
            )
            logger.info("Redis Checkpointer 初始化完成")

        # 构建中间件列表
        middleware = []

        # 对话摘要中间件：当消息数超过阈值时，自动用 LLM 生成摘要替换旧消息
        if self.summary_enabled:
            # 摘要模型：指定时使用独立模型，未指定时复用主模型
            if self.summary_model:
                summary_llm = ChatQwen(
                    model=self.summary_model,
                    api_key=config.dashscope_api_key,
                    temperature=0.3,  # 摘要任务用较低温度，保证稳定性
                )
            else:
                summary_llm = self.model

            summary_middleware = SummarizationMiddleware(
                model=summary_llm,
                trigger=("messages", self.summary_trigger_messages),
                keep=("messages", self.summary_keep_messages),
                summary_prompt=SUMMARY_PROMPT_ZH,
            )
            middleware.append(summary_middleware)
            logger.info(
                f"对话摘要中间件已启用: trigger={self.summary_trigger_messages} 条消息, "
                f"keep={self.summary_keep_messages} 条消息"
            )

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
            middleware=middleware,
        )

        self._agent_initialized = True

        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
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

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    async def _ensure_checkpoint_restored(self, session_id: str) -> None:
        """确保 Redis checkpoint 存在，若不存在则从 MySQL 恢复

        当 Redis TTL 过期导致 checkpoint 丢失时，从 MySQL 对话历史表
        加载历史消息并写回 Redis checkpointer，使 LangGraph 能恢复上下文。

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

        # Redis 中无数据，尝试从 MySQL 恢复
        logger.info(f"[会话 {session_id}] Redis checkpoint 不存在，尝试从 MySQL 恢复...")
        mysql_messages = await conversation_history_service.load_messages(session_id)
        if not mysql_messages:
            logger.info(f"[会话 {session_id}] MySQL 中也无历史记录")
            return

        # 将 MySQL 中的历史消息恢复到 Redis checkpointer
        # 构造一个初始 checkpoint 来存储这些消息
        from langgraph.checkpoint.base import Checkpoint
        from datetime import datetime, timezone
        import uuid

        checkpoint = Checkpoint(
            v=1,
            id=str(uuid.uuid4()),
            ts=datetime.now(timezone.utc).isoformat(),
            channel_values={"messages": mysql_messages},
            channel_versions={"messages": 1},
            versions_seen={},
            updated_channels=None,
        )
        metadata = {"source": "mysql_restore", "step": 0, "parents": {}}
        restore_config = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": "",
            }
        }

        await self.checkpointer.aput(
            restore_config, checkpoint, metadata, {"messages": 1}
        )

        # 设置 TTL
        if self.checkpointer.ttl > 0:
            key = self.checkpointer._key(session_id, "")
            await self.checkpointer.conn.expire(key, config.conversation_history_redis_ttl)

        logger.info(
            f"[会话 {session_id}] 已从 MySQL 恢复 {len(mysql_messages)} 条消息到 Redis "
            f"(TTL={config.conversation_history_redis_ttl}s)"
        )

    async def _sync_to_mysql(self, session_id: str, result: dict) -> None:
        """将对话结果推入消息队列，异步持久化到 MySQL

        在 Agent 执行完成后，将本轮对话的新消息序列化后推入消息队列，
        由后台消费者协程异步消费并写入 MySQL。非阻塞，不影响响应速度。

        Args:
            session_id: 会话 ID
            result: Agent 执行结果
        """
        if not config.conversation_history_enabled:
            return

        try:
            messages_result = result.get("messages", [])
            if not messages_result:
                return

            # 获取当前 MySQL 中已有的最大序号
            max_order = await conversation_history_service.get_max_order(session_id)
            start_order = max_order + 1

            # 推入消息队列（非阻塞）
            success = await conversation_history_service.enqueue_for_persist(
                session_id, messages_result, start_order=start_order
            )
            if success:
                logger.info(
                    f"[会话 {session_id}] 已将消息推入队列，等待异步持久化"
                )
            else:
                logger.warning(
                    f"[会话 {session_id}] 消息入队失败（已记录兜底日志）"
                )
        except Exception as e:
            # 入队失败不影响主流程
            logger.error(f"[会话 {session_id}] 推入消息队列失败: {e}")

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            # 确保 Redis checkpoint 存在（若过期则从 MySQL 恢复）
            await self._ensure_checkpoint_restored(session_id)

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 异步同步到 MySQL（不阻塞主流程）
            await self._sync_to_mysql(session_id, result)

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

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
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

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
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # 收集流式输出中的完整消息，用于后续同步到 MySQL
            collected_messages: list[BaseMessage] = []

            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            # 流式完成后，从 Redis checkpoint 获取完整消息并同步到 MySQL
            try:
                checkpoint_tuple = await self.checkpointer.aget_tuple(config_dict)
                if checkpoint_tuple:
                    all_messages = checkpoint_tuple.checkpoint.get(
                        "channel_values", {}
                    ).get("messages", [])
                    await self._sync_to_mysql(session_id, {"messages": all_messages})
            except Exception as sync_err:
                logger.warning(f"[会话 {session_id}] 流式完成后同步 MySQL 失败: {sync_err}")

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            self._log_exception_group(f"[会话 {session_id}] RAG Agent 查询失败（流式）", e)
            yield {
                "type": "error",
                "data": str(e)
            }
            raise

    async def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（优先从 Redis checkpointer 读取，Redis 无数据时从 MySQL 加载）

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

            # 2. Redis 无数据时，从 MySQL 加载
            if not messages and config.conversation_history_enabled:
                messages = await conversation_history_service.load_messages(session_id)
                if messages:
                    source = "mysql"
                    logger.info(
                        f"[会话 {session_id}] Redis 无数据，从 MySQL 加载了 {len(messages)} 条历史"
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

                # 识别摘要消息（由 SummarizationMiddleware 注入）
                if (
                    isinstance(msg, HumanMessage)
                    and getattr(msg, 'additional_kwargs', {}).get('lc_source') == 'summarization'
                ):
                    role = "summary"
                else:
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)

                # 提取时间戳（如果有的话）
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })

            logger.info(
                f"获取会话历史: {session_id}, 消息数量: {len(history)}, 来源: {source}"
            )
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    async def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（同时删除 Redis checkpoint 和 MySQL 对话历史）

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

            # 2. 删除 MySQL 对话历史
            if config.conversation_history_enabled:
                await conversation_history_service.delete_session(session_id)
                logger.info(f"已清除 MySQL 对话历史: {session_id}")

            logger.info(f"已清除会话历史: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

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

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出，摘要中间件默认开启
# 可通过参数自定义摘要行为：
#   summary_enabled=False          关闭摘要
#   summary_trigger_messages=10    触发摘要的消息数阈值
#   summary_keep_messages=6        摘要后保留最近消息数
#   summary_model="qwen-turbo"     使用独立模型做摘要
rag_agent_service = RagAgentService(streaming=True)
