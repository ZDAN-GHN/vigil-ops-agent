"""长期记忆服务单元测试

测试 LongTermMemoryService 的基本功能。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLongTermMemoryService:
    """长期记忆服务测试类"""

    @pytest.fixture
    def mock_session(self):
        """创建 mock 数据库会话"""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def service(self):
        """创建服务实例"""
        from app.services.long_term_memory_service import LongTermMemoryService
        return LongTermMemoryService()

    @pytest.mark.asyncio
    async def test_extract_features_from_summary_without_llm(self, service):
        """测试无 LLM 时的特征提取"""
        summary = "用户询问了 Kubernetes 集群的问题"

        # 没有配置 LLM
        service.llm = None
        features = await service.extract_features_from_summary(summary)

        assert features == {}

    @pytest.mark.asyncio
    async def test_extract_features_from_summary_with_llm(self, service):
        """测试有 LLM 时的特征提取"""
        summary = "用户是一名运维工程师，主要关注 Kubernetes 和 Docker 相关问题"

        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '''{
            "role": "运维工程师",
            "focus_areas": ["Kubernetes", "Docker"],
            "tech_stack": ["Kubernetes", "Docker"]
        }'''
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        service.llm = mock_llm

        features = await service.extract_features_from_summary(summary)

        assert "role" in features
        assert features["role"] == "运维工程师"
        assert "Kubernetes" in features.get("focus_areas", [])

    @pytest.mark.asyncio
    async def test_extract_features_invalid_json(self, service):
        """测试无效 JSON 的特征提取"""
        summary = "测试摘要"

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "这不是有效的 JSON"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        service.llm = mock_llm

        features = await service.extract_features_from_summary(summary)

        # 应该返回空字典而不是抛出异常
        assert features == {}

    @pytest.mark.asyncio
    async def test_build_user_context_empty(self, service):
        """测试空用户画像的上下文构建"""
        with patch("app.services.long_term_memory_service.mysql_manager") as mock_mysql:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            # 使用 async context manager
            async def get_session():
                yield mock_session

            mock_mysql.get_session = get_session

            context = await service.build_user_context("non-existent-user")
            assert context == ""

    @pytest.mark.asyncio
    async def test_build_user_context_with_profile(self, service):
        """测试有用户画像的上下文构建"""
        # Mock UserProfile
        mock_profile = MagicMock()
        mock_profile.features = {
            "role": "运维工程师",
            "focus_areas": ["Kubernetes", "Docker"],
            "tech_stack": ["Python", "Go"],
        }

        with patch("app.services.long_term_memory_service.mysql_manager") as mock_mysql:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_profile
            mock_session.execute = AsyncMock(return_value=mock_result)

            async def get_session():
                yield mock_session

            mock_mysql.get_session = get_session

            context = await service.build_user_context("test-user")

            assert "运维工程师" in context
            assert "Kubernetes" in context


class TestUserProfileModel:
    """用户画像模型测试"""

    def test_user_profile_creation(self):
        """测试用户画像创建"""
        from app.models.user_profile import UserProfile

        profile = UserProfile(
            user_id="test-user",
            features={"role": "运维工程师"},
            preferences={"response_style": "简洁"},
        )

        assert profile.user_id == "test-user"
        assert profile.features["role"] == "运维工程师"
        assert profile.preferences["response_style"] == "简洁"

    def test_conversation_summary_creation(self):
        """测试对话摘要创建"""
        from app.models.user_profile import ConversationSummary

        summary = ConversationSummary(
            session_id="test-session",
            user_id="test-user",
            summary="这是一条测试摘要",
            features_extracted={"role": "运维工程师"},
            message_count=10,
        )

        assert summary.session_id == "test-session"
        assert summary.user_id == "test-user"
        assert summary.message_count == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])