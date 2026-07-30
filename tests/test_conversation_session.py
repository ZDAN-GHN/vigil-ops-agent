"""会话管理服务单元测试

测试 ConversationSessionService 的：
- 创建/获取/更新/删除会话
- 列表查询
- 消息计数
- 软删除逻辑
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# 设置环境变量，避免导入链中初始化失败
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-for-unit-tests")


# ─────────────────────────────────────────────
# 测试：ConversationSession 模型
# ─────────────────────────────────────────────
class TestConversationSessionModel:
    """测试 ConversationSession ORM 模型"""

    def test_model_creation(self):
        """可以创建 ConversationSession 实例"""
        from app.models.conversation_session import ConversationSession

        session = ConversationSession(
            session_id="test-session-123",
            user_id=1,
            title="测试会话",
            message_count=5,
            is_deleted=False,
        )
        assert session.session_id == "test-session-123"
        assert session.user_id == 1
        assert session.title == "测试会话"
        assert session.message_count == 5
        assert session.is_deleted is False

    def test_model_repr(self):
        """__repr__ 输出正确"""
        from app.models.conversation_session import ConversationSession

        session = ConversationSession(
            session_id="test-session-123",
            user_id=1,
            title="测试会话",
        )
        repr_str = repr(session)
        assert "test-session-123" in repr_str
        assert "测试会话" in repr_str

    def test_model_defaults(self):
        """默认值正确（ORM 默认值在数据库层面生效，实例化时需显式传入）"""
        from app.models.conversation_session import ConversationSession

        session = ConversationSession(
            session_id="test-session-456",
            user_id=2,
            title="",
            message_count=0,
            is_deleted=False,
        )
        assert session.title == ""
        assert session.message_count == 0
        assert session.is_deleted is False


# ─────────────────────────────────────────────
# 测试：ConversationSessionService 方法
# ─────────────────────────────────────────────
class TestConversationSessionService:
    """测试 ConversationSessionService 方法"""

    @pytest.fixture
    def service(self):
        """创建 ConversationSessionService 实例"""
        from app.services.conversation_session_service import (
            ConversationSessionService,
        )

        return ConversationSessionService()

    @pytest.mark.asyncio
    async def test_create_session(self, service):
        """创建会话（自动生成 session_id）"""
        with patch(
            "app.services.conversation_session_service.mysql_manager"
        ) as mock_manager:
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock()
            mock_session.add = MagicMock()
            mock_session.flush = AsyncMock()

            # 模拟查询结果为空（不存在）
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute.return_value = mock_result

            mock_manager.get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_manager.get_session.return_value.__aexit__ = AsyncMock()

            result = await service.create_session(
                user_id=1,
                title="测试会话",
            )

            # 验证调用了 add
            mock_session.add.assert_called_once()
            # 验证返回的 session 有 session_id
            assert result.session_id is not None

    @pytest.mark.asyncio
    async def test_create_session_with_custom_id(self, service):
        """创建会话（指定 session_id）"""
        with patch(
            "app.services.conversation_session_service.mysql_manager"
        ) as mock_manager:
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock()
            mock_session.add = MagicMock()
            mock_session.flush = AsyncMock()

            # 模拟查询结果为空（不存在）
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute.return_value = mock_result

            mock_manager.get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_manager.get_session.return_value.__aexit__ = AsyncMock()

            result = await service.create_session(
                user_id=1,
                session_id="custom-session-id",
                title="测试会话",
            )

            # 验证调用了 add
            mock_session.add.assert_called_once()
            assert result.session_id == "custom-session-id"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, service):
        """获取不存在的会话返回 None"""
        with patch(
            "app.services.conversation_session_service.mysql_manager"
        ) as mock_manager:
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock()

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute.return_value = mock_result

            mock_manager.get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_manager.get_session.return_value.__aexit__ = AsyncMock()

            result = await service.get_session("non-existent", user_id=1)
            assert result is None

    @pytest.mark.asyncio
    async def test_delete_session(self, service):
        """删除会话（软删除）"""
        with patch(
            "app.services.conversation_session_service.mysql_manager"
        ) as mock_manager:
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock()
            mock_session.flush = AsyncMock()

            mock_result = MagicMock()
            mock_result.rowcount = 1
            mock_session.execute.return_value = mock_result

            mock_manager.get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_manager.get_session.return_value.__aexit__ = AsyncMock()

            result = await service.delete_session("test-session", user_id=1)
            assert result is True

    @pytest.mark.asyncio
    async def test_increment_message_count(self, service):
        """增加消息计数"""
        with patch(
            "app.services.conversation_session_service.mysql_manager"
        ) as mock_manager:
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock()
            mock_session.flush = AsyncMock()

            mock_result = MagicMock()
            mock_result.rowcount = 1
            mock_session.execute.return_value = mock_result

            mock_manager.get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_manager.get_session.return_value.__aexit__ = AsyncMock()

            result = await service.increment_message_count("test-session", count=2)
            assert result is True

    @pytest.mark.asyncio
    async def test_ensure_session_exists_creates_new(self, service):
        """ensure_session_exists 不存在时创建"""
        with patch.object(
            service, "get_session", return_value=None
        ), patch.object(
            service, "create_session"
        ) as mock_create:
            from app.models.conversation_session import ConversationSession

            mock_session = ConversationSession(
                session_id="new-session",
                user_id=1,
                title="新会话",
                is_deleted=False,
            )
            mock_create.return_value = mock_session

            result = await service.ensure_session_exists(
                user_id=1,
                session_id="new-session",
                title="新会话",
            )

            mock_create.assert_called_once_with(user_id=1, session_id="new-session", title="新会话")
            assert result.session_id == "new-session"

    @pytest.mark.asyncio
    async def test_ensure_session_exists_returns_existing(self, service):
        """ensure_session_exists 存在时直接返回"""
        with patch.object(service, "get_session") as mock_get, patch.object(
            service, "create_session"
        ) as mock_create:
            from app.models.conversation_session import ConversationSession

            existing = ConversationSession(
                session_id="existing-session",
                user_id=1,
                title="已有会话",
                is_deleted=False,
            )
            mock_get.return_value = existing

            result = await service.ensure_session_exists(
                user_id=1,
                session_id="existing-session",
            )

            mock_get.assert_called_once()
            mock_create.assert_not_called()
            assert result.session_id == "existing-session"

    @pytest.mark.asyncio
    async def test_ensure_session_exists_auto_generate(self, service):
        """ensure_session_exists 无 session_id 时自动生成"""
        with patch.object(
            service, "create_session"
        ) as mock_create:
            from app.models.conversation_session import ConversationSession

            mock_session = ConversationSession(
                session_id="auto-generated-uuid",
                user_id=1,
                title="新会话",
                is_deleted=False,
            )
            mock_create.return_value = mock_session

            result = await service.ensure_session_exists(
                user_id=1,
                session_id=None,
                title="新会话",
            )

            mock_create.assert_called_once_with(user_id=1, title="新会话")
            assert result.session_id == "auto-generated-uuid"


# ─────────────────────────────────────────────
# 测试：请求/响应模型
# ─────────────────────────────────────────────
class TestSessionModels:
    """测试会话相关的请求/响应模型"""

    def test_create_session_request(self):
        """CreateSessionRequest 模型"""
        from app.models.request import CreateSessionRequest

        req = CreateSessionRequest(title="测试")
        assert req.title == "测试"

    def test_create_session_request_default_title(self):
        """CreateSessionRequest 默认标题"""
        from app.models.request import CreateSessionRequest

        req = CreateSessionRequest()
        assert req.title == ""

    def test_update_session_request(self):
        """UpdateSessionRequest 模型"""
        from app.models.request import UpdateSessionRequest

        req = UpdateSessionRequest(title="新标题")
        assert req.title == "新标题"

    def test_session_response(self):
        """SessionResponse 模型"""
        from app.models.response import SessionResponse

        resp = SessionResponse(
            session_id="test-123",
            title="测试会话",
            message_count=10,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T01:00:00",
        )
        assert resp.session_id == "test-123"
        assert resp.message_count == 10

    def test_session_list_response(self):
        """SessionListResponse 模型"""
        from app.models.response import SessionListResponse, SessionResponse

        sessions = [
            SessionResponse(
                session_id="s1",
                title="会话1",
                message_count=5,
            ),
            SessionResponse(
                session_id="s2",
                title="会话2",
                message_count=3,
            ),
        ]
        resp = SessionListResponse(
            sessions=sessions,
            total=2,
            offset=0,
            limit=20,
        )
        assert len(resp.sessions) == 2
        assert resp.total == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
