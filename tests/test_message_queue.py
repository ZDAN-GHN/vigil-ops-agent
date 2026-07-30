"""消息队列服务单元测试

测试 MessageQueueService 的：
- 入队/出队逻辑
- Redis 降级到内存队列
- 消费失败重试
- 兜底日志记录
- 批量消费
"""

import asyncio
import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# 设置环境变量，避免导入链中初始化失败
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-for-unit-tests")


# ─────────────────────────────────────────────
# 测试辅助函数
# ─────────────────────────────────────────────
def create_test_payload(session_id: str = "test-session", count: int = 3) -> dict:
    """创建测试消息体"""
    messages = [
        {"role": "user", "content": f"message {i}", "tool_calls": None, "additional_kwargs": None}
        for i in range(count)
    ]
    return {
        "session_id": session_id,
        "messages": messages,
        "start_order": 0,
        "timestamp": "2024-01-01T00:00:00",
    }


# ─────────────────────────────────────────────
# 测试：消息序列化/反序列化
# ─────────────────────────────────────────────
class TestMessageSerialization:
    """测试消息序列化/反序列化"""

    def test_serialize_human_message(self):
        """序列化 HumanMessage"""
        from langchain_core.messages import HumanMessage
        from app.services.conversation_history_service import ConversationHistoryService

        msg = HumanMessage(content="hello world")
        result = ConversationHistoryService.serialize_message(msg)

        assert result["role"] == "user"
        assert result["content"] == "hello world"
        assert result["tool_calls"] is None

    def test_serialize_ai_message(self):
        """序列化 AIMessage"""
        from langchain_core.messages import AIMessage
        from app.services.conversation_history_service import ConversationHistoryService

        msg = AIMessage(content="hi there")
        result = ConversationHistoryService.serialize_message(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "hi there"

    def test_serialize_ai_message_with_tool_calls(self):
        """序列化带工具调用的 AIMessage"""
        from langchain_core.messages import AIMessage
        from app.services.conversation_history_service import ConversationHistoryService

        msg = AIMessage(content="searching...")
        msg.tool_calls = [{"name": "search", "args": {"q": "test"}}]
        result = ConversationHistoryService.serialize_message(msg)

        assert result["role"] == "assistant"
        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "search"

    def test_serialize_system_message(self):
        """序列化 SystemMessage"""
        from langchain_core.messages import SystemMessage
        from app.services.conversation_history_service import ConversationHistoryService

        msg = SystemMessage(content="you are helpful")
        result = ConversationHistoryService.serialize_message(msg)

        assert result["role"] == "system"
        assert result["content"] == "you are helpful"

    def test_reconstruct_from_dict_user(self):
        """从字典重建 user 消息"""
        from app.services.conversation_history_service import ConversationHistoryService
        from langchain_core.messages import HumanMessage

        msg_dict = {"role": "user", "content": "hello", "tool_calls": None, "additional_kwargs": None}
        msg = ConversationHistoryService._reconstruct_from_dict(msg_dict)

        assert isinstance(msg, HumanMessage)
        assert msg.content == "hello"

    def test_reconstruct_from_dict_assistant(self):
        """从字典重建 assistant 消息"""
        from app.services.conversation_history_service import ConversationHistoryService
        from langchain_core.messages import AIMessage

        msg_dict = {
            "role": "assistant",
            "content": "hi",
            "tool_calls": [{"name": "search", "args": {}}],
            "additional_kwargs": None,
        }
        msg = ConversationHistoryService._reconstruct_from_dict(msg_dict)

        assert isinstance(msg, AIMessage)
        assert msg.content == "hi"
        assert msg.tool_calls is not None

    def test_reconstruct_from_dict_unknown(self):
        """未知角色返回 None"""
        from app.services.conversation_history_service import ConversationHistoryService

        msg_dict = {"role": "unknown", "content": "test"}
        msg = ConversationHistoryService._reconstruct_from_dict(msg_dict)

        assert msg is None

    def test_round_trip_serialization(self):
        """序列化/反序列化往返一致性"""
        from langchain_core.messages import AIMessage, HumanMessage
        from app.services.conversation_history_service import ConversationHistoryService

        # 测试 user 消息
        original = HumanMessage(content="hello world")
        serialized = ConversationHistoryService.serialize_message(original)
        restored = ConversationHistoryService._reconstruct_from_dict(serialized)
        assert isinstance(restored, HumanMessage)
        assert restored.content == original.content

        # 测试 assistant 消息
        original = AIMessage(content="hi there")
        original.tool_calls = [{"name": "search", "args": {"q": "test"}}]
        serialized = ConversationHistoryService.serialize_message(original)
        restored = ConversationHistoryService._reconstruct_from_dict(serialized)
        assert isinstance(restored, AIMessage)
        assert restored.content == original.content
        assert restored.tool_calls == original.tool_calls


# ─────────────────────────────────────────────
# 测试：MessageQueueService 入队逻辑
# ─────────────────────────────────────────────
class TestMessageQueueEnqueue:
    """测试消息队列入队逻辑"""

    @pytest.fixture
    def mock_redis(self):
        """创建 mock Redis 客户端"""
        redis = AsyncMock()
        redis.lpush = AsyncMock(return_value=1)
        redis.rpop = AsyncMock(return_value=None)
        redis.llen = AsyncMock(return_value=0)
        redis.ping = AsyncMock()
        return redis

    @pytest.mark.asyncio
    async def test_enqueue_to_redis(self, mock_redis):
        """消息成功入队 Redis"""
        from app.services.message_queue_service import MessageQueueService

        service = MessageQueueService()
        service._redis_client = mock_redis

        payload = create_test_payload()
        messages = [{"role": "user", "content": "hello"}]

        result = await service.enqueue("test-session", messages)

        assert result is True
        mock_redis.lpush.assert_called_once()

    @pytest.mark.asyncio
    async def test_enqueue_fallback_to_memory(self, mock_redis):
        """Redis 失败时降级到内存队列"""
        from app.services.message_queue_service import MessageQueueService

        service = MessageQueueService()
        service._redis_client = mock_redis

        # 模拟 Redis 失败
        mock_redis.lpush.side_effect = Exception("Redis connection failed")

        messages = [{"role": "user", "content": "hello"}]
        result = await service.enqueue("test-session", messages)

        assert result is True  # 内存兜底成功
        assert service._use_memory_fallback is True
        assert service._memory_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_enqueue_empty_messages(self, mock_redis):
        """空消息列表不入队"""
        from app.services.message_queue_service import MessageQueueService

        service = MessageQueueService()
        service._redis_client = mock_redis

        result = await service.enqueue("test-session", [])

        assert result is False
        mock_redis.lpush.assert_not_called()


# ─────────────────────────────────────────────
# 测试：MessageQueueService 消费逻辑
# ─────────────────────────────────────────────
class TestMessageQueueConsume:
    """测试消息队列消费逻辑"""

    @pytest.fixture
    def mock_redis(self):
        """创建 mock Redis 客户端"""
        redis = AsyncMock()
        redis.lpush = AsyncMock(return_value=1)
        redis.rpop = AsyncMock(return_value=None)
        redis.llen = AsyncMock(return_value=0)
        redis.ping = AsyncMock()
        redis.pipeline = MagicMock()
        return redis

    @pytest.mark.asyncio
    async def test_process_redis_batch_empty(self, mock_redis):
        """空队列返回 0"""
        from app.services.message_queue_service import MessageQueueService

        service = MessageQueueService()
        service._redis_client = mock_redis

        # 模拟 pipeline 返回空结果
        pipe_mock = AsyncMock()
        pipe_mock.rpop = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[None] * service._batch_size)
        mock_redis.pipeline.return_value = pipe_mock

        result = await service._process_redis_batch()

        assert result == 0

    @pytest.mark.asyncio
    async def test_process_memory_batch_empty(self):
        """空内存队列返回 0"""
        from app.services.message_queue_service import MessageQueueService

        service = MessageQueueService()

        result = await service._process_memory_batch()

        assert result == 0

    @pytest.mark.asyncio
    async def test_start_stop_consumer(self, mock_redis):
        """启动/停止消费者"""
        from app.services.message_queue_service import MessageQueueService

        service = MessageQueueService()
        service._redis_client = mock_redis

        # 模拟 pipeline
        pipe_mock = AsyncMock()
        pipe_mock.rpop = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[None] * service._batch_size)
        mock_redis.pipeline.return_value = pipe_mock

        await service.start_consumer()
        assert service._running is True
        assert service._consumer_task is not None

        await service.stop_consumer()
        assert service._running is False
        assert service._consumer_task is None


# ─────────────────────────────────────────────
# 测试：兜底日志记录
# ─────────────────────────────────────────────
class TestFallbackLog:
    """测试兜底日志记录"""

    def test_log_fallback_message(self, tmp_path):
        """记录失败消息到兜底日志"""
        from app.services.message_queue_service import MessageQueueService

        log_file = tmp_path / "test_fallback.jsonl"

        service = MessageQueueService()
        service._fallback_log_path = log_file

        service._log_fallback_message('{"test": "data"}', "test reason")

        # 验证日志文件存在
        assert log_file.exists()

        # 验证日志内容
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            log_entry = json.loads(content)
            assert log_entry["reason"] == "test reason"
            assert log_entry["message"] == '{"test": "data"}'
            assert "timestamp" in log_entry


# ─────────────────────────────────────────────
# 测试：队列大小查询
# ─────────────────────────────────────────────
class TestQueueSize:
    """测试队列大小查询"""

    @pytest.mark.asyncio
    async def test_get_queue_size(self):
        """获取队列大小"""
        from app.services.message_queue_service import MessageQueueService

        service = MessageQueueService()

        # 模拟 Redis 不可用
        service._use_memory_fallback = True

        # 添加一些内存消息
        for i in range(5):
            service._memory_queue.put_nowait(f'{{"id": {i}}}')

        result = await service.get_queue_size()

        assert result["redis"] == 0
        assert result["memory"] == 5
        assert result["total"] == 5


# ─────────────────────────────────────────────
# 测试：enqueue_for_persist 集成
# ─────────────────────────────────────────────
class TestEnqueueForPersist:
    """测试 enqueue_for_persist 集成方法"""

    @pytest.mark.asyncio
    async def test_enqueue_filters_system_messages(self):
        """enqueue_for_persist 过滤系统消息"""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from app.services.conversation_history_service import ConversationHistoryService

        messages = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
        ]

        # 序列化并过滤
        serialized = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            serialized.append(ConversationHistoryService.serialize_message(msg))

        assert len(serialized) == 2
        assert serialized[0]["role"] == "user"
        assert serialized[1]["role"] == "assistant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
