"""对话历史持久化服务

负责将对话消息持久化到 MySQL，并在 Redis checkpoint 过期后恢复到 Redis。

核心功能：
1. save_messages() —— 将本轮对话消息增量写入 MySQL
2. load_messages() —— 从 MySQL 按 session_id 加载完整历史
3. restore_to_redis() —— 将 MySQL 中的历史恢复到 Redis checkpointer
4. delete_session() —— 删除指定会话的所有历史记录
"""

import json
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from loguru import logger
from sqlalchemy import delete, select, func

from app.config import config
from app.core.mysql_client import mysql_manager
from app.models.conversation_history import ConversationHistory


class ConversationHistoryService:
    """对话历史持久化服务

    提供 MySQL 持久化和 Redis 恢复能力，确保对话历史在 Redis TTL 过期后
    仍可从 MySQL 加载并恢复到 Redis checkpoint。
    """

    async def init_db(self) -> None:
        """初始化数据库表（创建 conversation_histories 表）"""
        from app.models.conversation_history import Base
        from app.models.user_profile import Base as UserProfileBase

        engine = await mysql_manager.get_engine()
        async with engine.begin() as conn:
            # 使用 user_profile 的 Base（共享 metadata）
            await conn.run_sync(UserProfileBase.metadata.create_all)
        logger.info("对话历史数据库表初始化完成")

    async def save_messages(
        self,
        session_id: str,
        messages: list[BaseMessage],
        start_order: int = 0,
    ) -> int:
        """将消息列表增量写入 MySQL

        Args:
            session_id: 会话 ID
            messages: 消息列表（LangChain BaseMessage 对象）
            start_order: 起始序号（增量写入时传入当前最大序号 + 1）

        Returns:
            int: 写入的消息数量
        """
        if not messages:
            return 0

        try:
            async with mysql_manager.get_session() as session:
                records = []
                order = start_order

                for msg in messages:
                    # 跳过系统消息（不持久化）
                    if isinstance(msg, SystemMessage):
                        continue

                    # 确定角色
                    role = self._get_role(msg)

                    # 提取工具调用信息
                    tool_calls_json = None
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        try:
                            tool_calls_json = json.dumps(
                                msg.tool_calls, ensure_ascii=False, default=str
                            )
                        except Exception:
                            tool_calls_json = None

                    # 提取额外元数据
                    metadata_json = None
                    additional_kwargs = getattr(msg, "additional_kwargs", {})
                    if additional_kwargs:
                        try:
                            metadata_json = json.dumps(
                                additional_kwargs, ensure_ascii=False, default=str
                            )
                        except Exception:
                            metadata_json = None

                    record = ConversationHistory(
                        session_id=session_id,
                        role=role,
                        content=msg.content if hasattr(msg, "content") else str(msg),
                        message_order=order,
                        tool_calls=tool_calls_json,
                        metadata_json=metadata_json,
                    )
                    records.append(record)
                    order += 1

                if records:
                    session.add_all(records)
                    await session.flush()

                logger.info(
                    f"保存对话历史: session={session_id}, "
                    f"写入 {len(records)} 条消息 (order={start_order}~{order - 1})"
                )
                return len(records)

        except Exception as e:
            logger.error(f"保存对话历史失败: session={session_id}, 错误: {e}")
            return 0

    async def get_max_order(self, session_id: str) -> int:
        """获取指定会话的最大消息序号

        Args:
            session_id: 会话 ID

        Returns:
            int: 最大序号，如果没有记录则返回 -1
        """
        try:
            async with mysql_manager.get_session() as session:
                result = await session.execute(
                    select(func.max(ConversationHistory.message_order)).where(
                        ConversationHistory.session_id == session_id
                    )
                )
                max_order = result.scalar_one_or_none()
                return max_order if max_order is not None else -1
        except Exception as e:
            logger.error(f"获取最大消息序号失败: session={session_id}, 错误: {e}")
            return -1

    async def load_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> list[BaseMessage]:
        """从 MySQL 加载对话历史

        Args:
            session_id: 会话 ID
            limit: 最大返回数量（None 表示不限制）

        Returns:
            list[BaseMessage]: 恢复的消息列表（LangChain 消息对象）
        """
        try:
            async with mysql_manager.get_session() as session:
                query = (
                    select(ConversationHistory)
                    .where(ConversationHistory.session_id == session_id)
                    .order_by(ConversationHistory.message_order)
                )
                if limit is not None:
                    query = query.limit(limit)

                result = await session.execute(query)
                records = result.scalars().all()

                messages = []
                for record in records:
                    msg = self._reconstruct_message(record)
                    if msg is not None:
                        messages.append(msg)

                logger.info(
                    f"从 MySQL 加载对话历史: session={session_id}, "
                    f"消息数量: {len(messages)}"
                )
                return messages

        except Exception as e:
            logger.error(f"加载对话历史失败: session={session_id}, 错误: {e}")
            return []

    async def has_history(self, session_id: str) -> bool:
        """检查指定会话是否有历史记录

        Args:
            session_id: 会话 ID

        Returns:
            bool: 是否有历史记录
        """
        try:
            async with mysql_manager.get_session() as session:
                result = await session.execute(
                    select(func.count(ConversationHistory.id)).where(
                        ConversationHistory.session_id == session_id
                    )
                )
                count = result.scalar_one()
                return count > 0
        except Exception as e:
            logger.error(f"检查历史记录失败: session={session_id}, 错误: {e}")
            return False

    async def delete_session(self, session_id: str) -> bool:
        """删除指定会话的所有历史记录

        Args:
            session_id: 会话 ID

        Returns:
            bool: 是否成功
        """
        try:
            async with mysql_manager.get_session() as session:
                result = await session.execute(
                    delete(ConversationHistory).where(
                        ConversationHistory.session_id == session_id
                    )
                )
                await session.flush()
                deleted_count = result.rowcount
                logger.info(
                    f"删除对话历史: session={session_id}, "
                    f"删除 {deleted_count} 条记录"
                )
                return True
        except Exception as e:
            logger.error(f"删除对话历史失败: session={session_id}, 错误: {e}")
            return False

    async def get_session_count(self, session_id: str) -> int:
        """获取指定会话的消息总数

        Args:
            session_id: 会话 ID

        Returns:
            int: 消息总数
        """
        try:
            async with mysql_manager.get_session() as session:
                result = await session.execute(
                    select(func.count(ConversationHistory.id)).where(
                        ConversationHistory.session_id == session_id
                    )
                )
                return result.scalar_one()
        except Exception as e:
            logger.error(f"获取会话消息数失败: session={session_id}, 错误: {e}")
            return 0

    # ── 消息序列化/反序列化 ──────────────────

    @staticmethod
    def serialize_message(msg: BaseMessage) -> dict:
        """将 LangChain 消息序列化为可 JSON 化的字典

        用于消息队列传输。

        Args:
            msg: LangChain 消息对象

        Returns:
            dict: 序列化后的字典
        """
        role = ConversationHistoryService._get_role(msg)
        content = msg.content if hasattr(msg, "content") else str(msg)

        # 工具调用信息
        tool_calls = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            try:
                tool_calls = json.loads(
                    json.dumps(msg.tool_calls, ensure_ascii=False, default=str)
                )
            except Exception:
                tool_calls = None

        # 额外元数据
        additional_kwargs = None
        raw_kwargs = getattr(msg, "additional_kwargs", {})
        if raw_kwargs:
            try:
                additional_kwargs = json.loads(
                    json.dumps(raw_kwargs, ensure_ascii=False, default=str)
                )
            except Exception:
                additional_kwargs = None

        return {
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "additional_kwargs": additional_kwargs,
        }

    @staticmethod
    def _reconstruct_from_dict(msg_dict: dict) -> Optional[BaseMessage]:
        """从字典反序列化为 LangChain 消息对象

        用于消息队列消费时重建消息。

        Args:
            msg_dict: 序列化后的字典

        Returns:
            BaseMessage 或 None
        """
        role = msg_dict.get("role", "unknown")
        content = msg_dict.get("content", "")

        if role == "user" or role == "summary":
            msg = HumanMessage(content=content)
            if role == "summary":
                additional_kwargs = msg_dict.get("additional_kwargs")
                if additional_kwargs:
                    msg.additional_kwargs = additional_kwargs
            return msg
        elif role == "assistant":
            msg = AIMessage(content=content)
            tool_calls = msg_dict.get("tool_calls")
            if tool_calls:
                msg.tool_calls = tool_calls
            return msg
        elif role == "system":
            return SystemMessage(content=content)
        else:
            logger.warning(f"未知消息角色: {role}, 跳过")
            return None

    async def enqueue_for_persist(
        self,
        session_id: str,
        messages: list[BaseMessage],
        start_order: int = 0,
    ) -> bool:
        """将消息推入消息队列，异步持久化到 MySQL

        非阻塞方法，消息会被序列化后推入 Redis List（或内存兜底队列），
        由后台消费者协程异步消费并写入 MySQL。

        Args:
            session_id: 会话 ID
            messages: 消息列表（LangChain BaseMessage 对象）
            start_order: 起始序号

        Returns:
            bool: 是否成功入队
        """
        if not config.conversation_history_enabled:
            return False

        if not messages:
            return False

        # 过滤系统消息并序列化
        serialized = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            serialized.append(self.serialize_message(msg))

        if not serialized:
            return False

        # 推入消息队列
        from app.services.message_queue_service import message_queue_service

        return await message_queue_service.enqueue(
            session_id, serialized, start_order=start_order
        )

    # ── 内部辅助方法 ──────────────────────────

    @staticmethod
    def _get_role(msg: BaseMessage) -> str:
        """获取消息角色"""
        if isinstance(msg, SystemMessage):
            return "system"
        elif isinstance(msg, HumanMessage):
            # 检查是否是摘要消息
            additional_kwargs = getattr(msg, "additional_kwargs", {})
            if additional_kwargs.get("lc_source") == "summarization":
                return "summary"
            return "user"
        elif isinstance(msg, AIMessage):
            return "assistant"
        else:
            return "unknown"

    @staticmethod
    def _reconstruct_message(record: ConversationHistory) -> Optional[BaseMessage]:
        """从数据库记录重建 LangChain 消息对象

        Args:
            record: 数据库记录

        Returns:
            BaseMessage 或 None
        """
        role = record.role
        content = record.content

        if role == "user" or role == "summary":
            msg = HumanMessage(content=content)
            # 恢复摘要标记
            if role == "summary" and record.metadata_json:
                try:
                    metadata = json.loads(record.metadata_json)
                    msg.additional_kwargs = metadata
                except Exception:
                    pass
            return msg
        elif role == "assistant":
            msg = AIMessage(content=content)
            # 恢复工具调用信息
            if record.tool_calls:
                try:
                    msg.tool_calls = json.loads(record.tool_calls)
                except Exception:
                    pass
            return msg
        elif role == "system":
            return SystemMessage(content=content)
        else:
            logger.warning(f"未知消息角色: {role}, 跳过")
            return None


# 全局单例
conversation_history_service = ConversationHistoryService()
