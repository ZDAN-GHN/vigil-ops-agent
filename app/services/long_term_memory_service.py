"""长期记忆服务

负责管理跨会话的用户画像和对话摘要历史。
主要功能：
1. 从对话摘要中提取用户特征
2. 存储和更新用户画像
3. 加载用户画像用于上下文增强
"""
from textwrap import dedent

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from sqlalchemy import select

from app.core.mysql_client import mysql_manager
from app.models.user_profile import ConversationSummary, UserFeatures, UserProfile


class LongTermMemoryService:
    """长期记忆服务

    管理用户画像和对话摘要历史。
    """

    def __init__(self, llm=None) -> None:
        """
        初始化长期记忆服务

        Args:
            llm: 用于特征提取的 LLM 实例（可选）
        """
        self.llm = llm

    async def init_db(self) -> None:
        """初始化数据库表"""
        from app.models.user_profile import Base

        engine = await mysql_manager.get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("长期记忆数据库表初始化完成")

    async def get_user_profile(self, user_id: str) -> UserProfile | None:
        """
        获取用户画像

        Args:
            user_id: 用户 ID

        Returns:
            UserProfile 或 None
        """
        async with mysql_manager.get_session() as session:
            result = await session.execute(
                select(UserProfile).where(UserProfile.user_id == user_id)
            )
            return result.scalar_one_or_none()

    async def create_or_update_profile(
            self,
            user_id: str,
            features: dict | None = None,
            preferences: dict | None = None,
    ) -> UserProfile:
        """
        创建或更新用户画像

        Args:
            user_id: 用户 ID
            features: 用户特征
            preferences: 用户偏好

        Returns:
            更新后的 UserProfile
        """
        async with mysql_manager.get_session() as session:
            result = await session.execute(
                select(UserProfile).where(UserProfile.user_id == user_id)
            )
            profile = result.scalar_one_or_none()

            if profile is None:
                # 创建新用户画像
                profile = UserProfile(
                    user_id=user_id,
                    features=features or {},
                    preferences=preferences or {},
                )
                session.add(profile)
                logger.info(f"创建新用户画像: {user_id}")
            else:
                # 更新现有画像（合并特征）
                if features:
                    merged_features = {**profile.features, **features}
                    profile.features = merged_features
                if preferences:
                    merged_prefs = {**profile.preferences, **preferences}
                    profile.preferences = merged_prefs
                logger.info(f"更新用户画像: {user_id}")

            await session.flush()
            return profile

    async def save_conversation_summary(
            self,
            session_id: str,
            summary: str,
            user_id: str | None = None,
            features_extracted: dict | None = None,
            message_count: int = 0,
    ) -> ConversationSummary:
        """
        保存对话摘要记录

        Args:
            session_id: 会话 ID
            summary: 摘要内容
            user_id: 用户 ID（可选）
            features_extracted: 提取的特征
            message_count: 原始消息数量

        Returns:
            保存的 ConversationSummary
        """
        async with mysql_manager.get_session() as session:
            record = ConversationSummary(
                session_id=session_id,
                user_id=user_id,
                summary=summary,
                features_extracted=features_extracted or {},
                message_count=message_count,
            )
            session.add(record)
            await session.flush()
            logger.info(f"保存对话摘要: session={session_id}, user={user_id}")
            return record

    async def extract_features_from_summary(self, summary: str) -> dict:
        """
        从对话摘要中提取用户特征

        使用 LangChain LCEL 管道 + with_structured_output 实现结构化输出，
        与 planner.py / replanner.py 保持一致的模式。

        Args:
            summary: 对话摘要

        Returns:
            提取的特征字典
        """
        if self.llm is None:
            logger.warning("未配置 LLM，跳过特征提取")
            return {}

        try:
            # 构建 LCEL 管道：提示词模板 → LLM 结构化输出
            # 从摘要中提取用户特征的提示词模板
            feature_extraction_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        dedent("""
                            你是一个用户画像分析专家。请从对话摘要中提取用户特征信息，覆盖以下维度（如果摘要中有相关内容）：
                            
                            1. 用户角色/职位（如：运维工程师、开发工程师、架构师）
                            2. 关注领域（如：Kubernetes、数据库、网络、安全）
                            3. 技术栈偏好（如：使用的工具、语言、框架）
                            4. 工作习惯（如：偏好简洁回答、喜欢详细解释）
                            5. 常见问题类型（如：故障排查、性能优化、架构设计）
                            
                            输出约束：
                            - 严格按照结构化格式输出，无法提取的维度留空
                            - 未按照约束执行的输出视作任务执行失败
                        """).strip(),
                    ),
                    ("human", "{summary}"),
                ]
            )
            feature_extraction_chain = feature_extraction_prompt | self.llm.with_structured_output(UserFeatures)
            result: UserFeatures = await feature_extraction_chain.ainvoke({"summary": summary})

            # 转为字典，排除默认空值字段以保持紧凑
            features = result.model_dump(exclude_defaults=True)
            logger.info(f"提取用户特征: {list(features.keys())}")
            return features

        except Exception as e:
            logger.error(f"提取用户特征失败: {e}")
            return {}

    async def process_summary(
            self,
            session_id: str,
            summary: str,
            user_id: str | None = None,
            message_count: int = 0,
    ) -> dict:
        """
        处理对话摘要：提取特征并更新用户画像

        Args:
            session_id: 会话 ID
            summary: 摘要内容
            user_id: 用户 ID（可选）
            message_count: 原始消息数量

        Returns:
            提取的特征字典
        """
        # 提取特征
        features = await self.extract_features_from_summary(summary)

        # 保存摘要记录
        await self.save_conversation_summary(
            session_id=session_id,
            summary=summary,
            user_id=user_id,
            features_extracted=features,
            message_count=message_count,
        )

        # 更新用户画像
        if user_id and features:
            await self.create_or_update_profile(
                user_id=user_id,
                features=features,
            )

        return features

    async def build_user_context(self, user_id: str) -> str:
        """
        构建用户上下文（用于注入系统提示词）

        Args:
            user_id: 用户 ID

        Returns:
            用户上下文描述字符串
        """
        profile = await self.get_user_profile(user_id)
        if profile is None:
            return ""

        parts = []

        # 角色信息
        role = profile.features.get("role")
        if role:
            parts.append(f"用户角色：{role}")

        # 关注领域
        focus_areas = profile.features.get("focus_areas", [])
        if focus_areas:
            parts.append(f"关注领域：{', '.join(focus_areas)}")

        # 技术栈
        tech_stack = profile.features.get("tech_stack", [])
        if tech_stack:
            parts.append(f"技术栈：{', '.join(tech_stack)}")

        # 偏好
        preferences = profile.features.get("preferences", {})
        if preferences:
            pref_str = ", ".join(f"{k}={v}" for k, v in preferences.items())
            parts.append(f"偏好：{pref_str}")

        if not parts:
            return ""

        return "【用户画像】\n" + "\n".join(parts)

    async def get_recent_summaries(self, user_id: str, limit: int = 5) -> list[ConversationSummary]:
        """
        获取用户最近的对话摘要

        Args:
            user_id: 用户 ID
            limit: 最大返回数量

        Returns:
            摘要记录列表
        """
        async with mysql_manager.get_session() as session:
            result = await session.execute(
                select(ConversationSummary)
                .where(ConversationSummary.user_id == user_id)
                .order_by(ConversationSummary.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())


# 全局单例（延迟初始化 LLM）
long_term_memory_service = LongTermMemoryService()
