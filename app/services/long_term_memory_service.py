"""长期记忆服务

负责管理跨会话的用户画像和对话摘要历史。
主要功能：
1. 从对话摘要中提取用户特征
2. 存储和更新用户画像
3. 加载用户画像用于上下文增强
"""

from typing import Optional

from loguru import logger
from sqlalchemy import select

from app.core.mysql_client import mysql_manager
from app.models.user_profile import ConversationSummary, UserProfile

# 从摘要中提取用户特征的提示词
FEATURE_EXTRACTION_PROMPT = """请从以下对话摘要中提取用户特征信息。

对话摘要：
{summary}

请分析并提取以下维度的信息（如果摘要中有相关内容）：
1. 用户角色/职位（如：运维工程师、开发工程师、架构师）
2. 关注领域（如：Kubernetes、数据库、网络、安全）
3. 技术栈偏好（如：使用的工具、语言、框架）
4. 工作习惯（如：偏好简洁回答、喜欢详细解释）
5. 常见问题类型（如：故障排查、性能优化、架构设计）

请以 JSON 格式返回，格式如下：
{{
    "role": "用户角色",
    "focus_areas": ["关注领域1", "关注领域2"],
    "tech_stack": ["技术1", "技术2"],
    "preferences": {{"response_style": "简洁/详细", ...}},
    "common_issues": ["问题类型1", "问题类型2"]
}}

只返回 JSON，不要其他内容。如果某个维度无法提取，可以省略或返回空值。"""


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

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
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
        features: Optional[dict] = None,
        preferences: Optional[dict] = None,
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
        user_id: Optional[str] = None,
        features_extracted: Optional[dict] = None,
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

        Args:
            summary: 对话摘要

        Returns:
            提取的特征字典
        """
        if self.llm is None:
            logger.warning("未配置 LLM，跳过特征提取")
            return {}

        try:
            prompt = FEATURE_EXTRACTION_PROMPT.format(summary=summary)
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()

            # 尝试解析 JSON
            import json
            # 移除可能的 markdown 代码块标记
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            features = json.loads(content)
            logger.info(f"提取用户特征: {list(features.keys())}")
            return features

        except json.JSONDecodeError as e:
            logger.warning(f"解析特征 JSON 失败: {e}")
            return {}
        except Exception as e:
            logger.error(f"提取用户特征失败: {e}")
            return {}

    async def process_summary(
        self,
        session_id: str,
        summary: str,
        user_id: Optional[str] = None,
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

    async def get_recent_summaries(
        self, user_id: str, limit: int = 5
    ) -> list[ConversationSummary]:
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
