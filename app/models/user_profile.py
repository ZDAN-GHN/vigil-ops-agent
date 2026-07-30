"""用户画像模型

用于存储跨会话的长期记忆，包括用户特征、偏好等信息。
"""

from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import JSON, DateTime, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy 基类"""

    pass


class UserProfile(Base):
    """用户画像表

    存储用户的长期记忆信息，包括：
    - 用户特征画像（从对话摘要中提取）
    - 偏好设置
    - 历史交互模式

    Attributes:
        user_id: 用户唯一标识（主键）
        features: 用户特征 JSON，如 {"role": "运维工程师", "focus": "Kubernetes"}
        preferences: 用户偏好 JSON，如 {"response_style": "简洁"}
        summary_count: 累计摘要次数
        last_summary_at: 最后一次摘要时间
        created_at: 创建时间
        updated_at: 更新时间
    """

    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True, comment="用户ID")
    features: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False, comment="用户特征画像"
    )
    preferences: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False, comment="用户偏好设置"
    )
    summary_count: Mapped[int] = mapped_column(default=0, comment="累计摘要次数")
    last_summary_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="最后一次摘要时间"
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, comment="备注信息")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间"
    )

    def __repr__(self) -> str:
        return f"<UserProfile(user_id={self.user_id}, features={self.features})>"


class ConversationSummary(Base):
    """对话摘要历史表

    存储每次对话摘要的记录，用于追溯和分析。

    Attributes:
        id: 自增主键
        session_id: 会话 ID
        user_id: 用户 ID（可选，用于关联用户画像）
        summary: 摘要内容
        features_extracted: 从摘要中提取的特征
        message_count: 原始消息数量
        created_at: 创建时间
    """

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True, comment="会话ID")
    user_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True, comment="用户ID"
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False, comment="摘要内容")
    features_extracted: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False, comment="提取的特征"
    )
    message_count: Mapped[int] = mapped_column(default=0, comment="原始消息数量")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), comment="创建时间"
    )

    def __repr__(self) -> str:
        return f"<ConversationSummary(session_id={self.session_id}, user_id={self.user_id})>"


class UserFeatures(BaseModel):
    """用户特征结构化输出模型

    用于 LLM 结构化输出，从对话摘要中提取用户特征。
    所有字段设置默认值，兼容"无法提取时省略"的场景。

    Attributes:
        role: 用户角色/职位（如：运维工程师、开发工程师、架构师）
        focus_areas: 关注领域列表（如：Kubernetes、数据库、网络、安全）
        tech_stack: 技术栈偏好列表（如：使用的工具、语言、框架）
        preferences: 工作习惯/偏好字典（如：response_style="简洁"）
        common_issues: 常见问题类型列表（如：故障排查、性能优化、架构设计）
    """

    role: str = Field(
        default="",
        description="用户角色/职位，如：运维工程师、开发工程师、架构师",
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="关注领域列表，如：Kubernetes、数据库、网络、安全",
    )
    tech_stack: list[str] = Field(
        default_factory=list,
        description="技术栈偏好列表，如：使用的工具、语言、框架",
    )
    preferences: dict[str, str] = Field(
        default_factory=dict,
        description="工作习惯/偏好字典，如：response_style='简洁'",
    )
    common_issues: list[str] = Field(
        default_factory=list,
        description="常见问题类型列表，如：故障排查、性能优化、架构设计",
    )
