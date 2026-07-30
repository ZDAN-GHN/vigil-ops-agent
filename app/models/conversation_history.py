"""对话历史模型

用于存储完整的对话历史消息，作为 Redis checkpoint 的 MySQL 持久化备份。
当 Redis 中的 checkpoint 过期后，可从 MySQL 恢复对话上下文到 Redis。

设计思路：
- 每条消息单独一行存储（便于增量写入和查询）
- 通过 session_id + message_order 排序恢复完整对话序列
- tool_calls 以 JSON 格式存储（可选，用于调试和回放）
"""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.models.user_profile import Base


class ConversationHistory(Base):
    """对话历史表

    存储每轮对话的完整消息记录，用于：
    1. Redis checkpoint 过期后从 MySQL 恢复对话上下文
    2. 查看会话历史记录（不依赖 Redis）
    3. 长期对话数据分析

    Attributes:
        id: 自增主键
        session_id: 会话 ID（即 thread_id）
        role: 消息角色（user / assistant / system / summary）
        content: 消息内容
        message_order: 消息在会话中的顺序（从 0 开始递增）
        tool_calls: 工具调用信息 JSON（可选）
        metadata_json: 额外元数据 JSON（可选）
        created_at: 创建时间
    """

    __tablename__ = "conversation_histories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True, comment="会话ID（thread_id）"
    )
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="消息角色: user/assistant/system/summary"
    )
    content: Mapped[str] = mapped_column(
        Text, nullable=False, comment="消息内容"
    )
    message_order: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="消息在会话中的顺序（从 0 开始）"
    )
    tool_calls: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="工具调用信息 JSON"
    )
    metadata_json: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="额外元数据 JSON"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), comment="创建时间"
    )

    # 复合索引：加速按 session_id 查询并按顺序排列
    __table_args__ = (
        Index("ix_session_order", "session_id", "message_order"),
    )

    def __repr__(self) -> str:
        return (
            f"<ConversationHistory(session_id={self.session_id}, "
            f"role={self.role}, order={self.message_order})>"
        )
