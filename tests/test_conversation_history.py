"""对话历史持久化服务单元测试

测试 ConversationHistoryService 的：
- MySQL 读写操作
- 消息序列化/反序列化
- 增量写入逻辑
- 会话删除
- Redis fallback 恢复逻辑
"""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# 设置环境变量，避免导入链中 DashScopeEmbeddings 初始化失败
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-for-unit-tests")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.models.conversation_history import ConversationHistory


# ─────────────────────────────────────────────
# 测试辅助函数
# ─────────────────────────────────────────────
def create_mock_record(
    session_id: str = "test-session",
    role: str = "user",
    content: str = "hello",
    message_order: int = 0,
    tool_calls: str | None = None,
    metadata_json: str | None = None,
) -> ConversationHistory:
    """创建测试用的数据库记录"""
    record = ConversationHistory(
        session_id=session_id,
        role=role,
        content=content,
        message_order=message_order,
        tool_calls=tool_calls,
        metadata_json=metadata_json,
    )
    # 手动设置 id（模拟数据库自增）
    record.id = message_order + 1
    return record


# ─────────────────────────────────────────────
# 测试：ConversationHistory 模型
# ─────────────────────────────────────────────
class TestConversationHistoryModel:
    """测试 ConversationHistory ORM 模型"""

    def test_model_creation(self):
        """可以创建 ConversationHistory 实例"""
        record = create_mock_record()
        assert record.session_id == "test-session"
        assert record.role == "user"
        assert record.content == "hello"
        assert record.message_order == 0

    def test_model_with_tool_calls(self):
        """可以存储工具调用信息"""
        tool_calls = json.dumps([{"name": "search", "args": {"q": "test"}}])
        record = create_mock_record(
            role="assistant",
            content="searching...",
            tool_calls=tool_calls,
        )
        assert record.tool_calls == tool_calls
        parsed = json.loads(record.tool_calls)
        assert parsed[0]["name"] == "search"

    def test_model_repr(self):
        """__repr__ 输出正确"""
        record = create_mock_record()
        repr_str = repr(record)
        assert "test-session" in repr_str
        assert "user" in repr_str


# ─────────────────────────────────────────────
# 测试：ConversationHistoryService 单元
# ─────────────────────────────────────────────
class TestConversationHistoryService:
    """测试 ConversationHistoryService 方法"""

    @pytest.fixture
    def service(self):
        """创建 ConversationHistoryService 实例"""
        from app.services.conversation_history_service import (
            ConversationHistoryService,
        )

        return ConversationHistoryService()

    def test_get_role_user(self, service):
        """识别 user 角色"""
        msg = HumanMessage(content="hello")
        role = service._get_role(msg)
        assert role == "user"

    def test_get_role_assistant(self, service):
        """识别 assistant 角色"""
        msg = AIMessage(content="hi")
        role = service._get_role(msg)
        assert role == "assistant"

    def test_get_role_system(self, service):
        """识别 system 角色"""
        msg = SystemMessage(content="you are helpful")
        role = service._get_role(msg)
        assert role == "system"

    def test_get_role_summary(self, service):
        """识别 summary 角色"""
        msg = HumanMessage(
            content="summary...",
            additional_kwargs={"lc_source": "summarization"},
        )
        role = service._get_role(msg)
        assert role == "summary"

    def test_reconstruct_user_message(self, service):
        """从数据库记录重建 user 消息"""
        record = create_mock_record(role="user", content="hello")
        msg = service._reconstruct_message(record)
        assert isinstance(msg, HumanMessage)
        assert msg.content == "hello"

    def test_reconstruct_assistant_message(self, service):
        """从数据库记录重建 assistant 消息"""
        record = create_mock_record(role="assistant", content="hi there")
        msg = service._reconstruct_message(record)
        assert isinstance(msg, AIMessage)
        assert msg.content == "hi there"

    def test_reconstruct_system_message(self, service):
        """从数据库记录重建 system 消息"""
        record = create_mock_record(role="system", content="you are helpful")
        msg = service._reconstruct_message(record)
        assert isinstance(msg, SystemMessage)
        assert msg.content == "you are helpful"

    def test_reconstruct_summary_message(self, service):
        """从数据库记录重建 summary 消息"""
        record = create_mock_record(
            role="summary",
            content="summary content",
            metadata_json=json.dumps({"lc_source": "summarization"}),
        )
        msg = service._reconstruct_message(record)
        assert isinstance(msg, HumanMessage)
        assert msg.content == "summary content"

    def test_reconstruct_unknown_role(self, service):
        """未知角色返回 None"""
        record = create_mock_record(role="unknown")
        msg = service._reconstruct_message(record)
        assert msg is None

    def test_reconstruct_with_tool_calls(self, service):
        """重建带有工具调用的 assistant 消息"""
        tool_calls_data = [{"name": "search", "args": {"q": "test"}}]
        record = create_mock_record(
            role="assistant",
            content="searching...",
            tool_calls=json.dumps(tool_calls_data),
        )
        msg = service._reconstruct_message(record)
        assert isinstance(msg, AIMessage)
        assert hasattr(msg, "tool_calls")
        assert msg.tool_calls == tool_calls_data


# ─────────────────────────────────────────────
# 测试：_ensure_checkpoint_restored 逻辑
# ─────────────────────────────────────────────
class TestCheckpointRestore:
    """测试 Redis checkpoint 从 MySQL 恢复逻辑"""

    @pytest.mark.asyncio
    async def test_restore_skips_when_redis_has_data(self):
        """Redis 有数据时不触发恢复"""
        from app.services.rag_agent_service import RagAgentService

        service = RagAgentService.__new__(RagAgentService)
        service.checkpointer = AsyncMock()
        service.checkpointer.aget_tuple = AsyncMock(return_value=MagicMock())

        with patch(
            "app.services.rag_agent_service.config"
        ) as mock_config:
            mock_config.conversation_history_enabled = True
            await service._ensure_checkpoint_restored("test-session")

        # 验证没有调用 conversation_history_service
        # （通过检查 checkpointer.aput 没有被调用）
        service.checkpointer.aput.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_skips_when_disabled(self):
        """功能关闭时不触发恢复"""
        from app.services.rag_agent_service import RagAgentService

        service = RagAgentService.__new__(RagAgentService)
        service.checkpointer = AsyncMock()

        with patch(
            "app.services.rag_agent_service.config"
        ) as mock_config:
            mock_config.conversation_history_enabled = False
            await service._ensure_checkpoint_restored("test-session")

        # 验证没有检查 Redis
        service.checkpointer.aget_tuple.assert_not_called()


# ─────────────────────────────────────────────
# 测试：_sync_to_mysql 逻辑
# ─────────────────────────────────────────────
class TestSyncToMySQL:
    """测试对话结果同步到 MySQL 的逻辑"""

    @pytest.mark.asyncio
    async def test_sync_skips_when_disabled(self):
        """功能关闭时不同步"""
        from app.services.rag_agent_service import RagAgentService

        service = RagAgentService.__new__(RagAgentService)

        with patch(
            "app.services.rag_agent_service.config"
        ) as mock_config:
            mock_config.conversation_history_enabled = False
            await service._sync_to_mysql("test-session", {"messages": []})

    @pytest.mark.asyncio
    async def test_sync_skips_empty_messages(self):
        """空消息列表不同步"""
        from app.services.rag_agent_service import RagAgentService

        service = RagAgentService.__new__(RagAgentService)

        with patch(
            "app.services.rag_agent_service.config"
        ) as mock_config:
            mock_config.conversation_history_enabled = True
            # 不应抛出异常
            await service._sync_to_mysql("test-session", {"messages": []})


# ─────────────────────────────────────────────
# 测试：clear_session 同时删除 MySQL
# ─────────────────────────────────────────────
class TestClearSession:
    """测试清空会话同时删除 MySQL 数据"""

    @pytest.mark.asyncio
    async def test_clear_deletes_both_redis_and_mysql(self):
        """清空会话时同时删除 Redis 和 MySQL 数据"""
        from app.services.rag_agent_service import RagAgentService

        service = RagAgentService.__new__(RagAgentService)
        service.checkpointer = AsyncMock()
        service.checkpointer.adelete_thread = AsyncMock()

        with patch(
            "app.services.rag_agent_service.config"
        ) as mock_config, patch(
            "app.services.rag_agent_service.conversation_history_service"
        ) as mock_history:
            mock_config.conversation_history_enabled = True
            mock_history.delete_session = AsyncMock(return_value=True)

            result = await service.clear_session("test-session")

            assert result is True
            service.checkpointer.adelete_thread.assert_called_once_with("test-session")
            mock_history.delete_session.assert_called_once_with("test-session")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
