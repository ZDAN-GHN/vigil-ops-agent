"""Redis Checkpointer 单元测试

测试新架构（BaseRedisSaver / RedisSaver / AsyncRedisSaver）的：
- 序列化/反序列化往返一致性
- Key 构造逻辑
- Blob dump/load
- Writes dump/load
- metadata 过滤
- 版本管理
- 完整 put -> get 流程（使用 mock）
- 单 Hash 合并模型验证
"""

import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


# 全局 serde 实例
_serde = JsonPlusSerializer()


# ─────────────────────────────────────────────
# 测试辅助函数
# ─────────────────────────────────────────────
def create_test_checkpoint(checkpoint_id: str = "test-cp-001") -> Checkpoint:
    """创建测试用的 checkpoint（包含真实的 BaseMessage 对象）"""
    return Checkpoint(
        v=1,
        id=checkpoint_id,
        ts="2024-01-01T00:00:00+00:00",
        channel_values={
            "messages": [
                SystemMessage(content="you are a helpful assistant"),
                HumanMessage(content="hello"),
                AIMessage(content="hi there"),
            ]
        },
        channel_versions={"messages": 1},
        versions_seen={},
        updated_channels=None,
    )


def create_test_metadata() -> dict:
    """创建测试用的 metadata"""
    return {"source": "loop", "step": 0, "parents": {}}


def _serialize_to_redis(obj) -> tuple[str, str]:
    """模拟序列化逻辑"""
    type_tag, raw_bytes = _serde.dumps_typed(obj)
    return type_tag, base64.b64encode(raw_bytes).decode("utf-8")


def _deserialize_from_redis(type_tag: str, b64_data: str):
    """模拟反序列化逻辑"""
    raw_bytes = base64.b64decode(b64_data)
    return _serde.loads_typed((type_tag, raw_bytes))


# ─────────────────────────────────────────────
# 测试：序列化/反序列化往返一致性
# ─────────────────────────────────────────────
class TestSerdeRoundTrip:
    """测试 serde 序列化/反序列化的往返一致性"""

    def test_system_message_round_trip(self):
        """SystemMessage 可以正确序列化并反序列化"""
        msg = SystemMessage(content="you are a helpful assistant")
        type_tag, b64 = _serialize_to_redis(msg)
        restored = _deserialize_from_redis(type_tag, b64)
        assert isinstance(restored, SystemMessage)
        assert restored.content == msg.content

    def test_human_message_round_trip(self):
        """HumanMessage 可以正确序列化并反序列化"""
        msg = HumanMessage(content="hello world")
        type_tag, b64 = _serialize_to_redis(msg)
        restored = _deserialize_from_redis(type_tag, b64)
        assert isinstance(restored, HumanMessage)
        assert restored.content == msg.content

    def test_ai_message_round_trip(self):
        """AIMessage 可以正确序列化并反序列化"""
        msg = AIMessage(content="hi there!")
        type_tag, b64 = _serialize_to_redis(msg)
        restored = _deserialize_from_redis(type_tag, b64)
        assert isinstance(restored, AIMessage)
        assert restored.content == msg.content

    def test_message_list_round_trip(self):
        """消息列表可以正确序列化并反序列化"""
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
        ]
        type_tag, b64 = _serialize_to_redis(messages)
        restored = _deserialize_from_redis(type_tag, b64)
        assert len(restored) == 3
        assert isinstance(restored[0], SystemMessage)
        assert isinstance(restored[1], HumanMessage)
        assert isinstance(restored[2], AIMessage)

    def test_checkpoint_structure_round_trip(self):
        """checkpoint 结构可以正确序列化"""
        checkpoint = create_test_checkpoint()
        c = checkpoint.copy()
        c.pop("channel_values", {})
        type_tag, b64 = _serialize_to_redis(c)
        restored = _deserialize_from_redis(type_tag, b64)
        assert restored["id"] == "test-cp-001"
        assert "channel_versions" in restored


# ─────────────────────────────────────────────
# 测试：Key 构造（单 Hash 模型）
# ─────────────────────────────────────────────
class TestKeyConstruction:
    """测试 Redis Key 构造逻辑"""

    def test_single_hash_key(self):
        """每个 thread+ns 只有一个 Hash key"""
        from app.core.redis_checkpointer import BaseRedisSaver

        key = BaseRedisSaver._key("thread-1", "ns-1")
        assert key == "ckp:thread-1:ns-1"

    def test_empty_ns(self):
        """空 namespace 的 key 构造"""
        from app.core.redis_checkpointer import BaseRedisSaver

        key = BaseRedisSaver._key("thread-1", "")
        assert key == "ckp:thread-1:"

    def test_cp_field(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        field = BaseRedisSaver._cp_field("cp-001")
        assert field == "c:cp-001"

    def test_blob_field(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        field = BaseRedisSaver._blob_field("messages", "1")
        assert field == "b:messages:1"

    def test_write_field(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        field = BaseRedisSaver._write_field("cp-001", "task-1", 0)
        assert field == "w:cp-001:task-1:0"


# ─────────────────────────────────────────────
# 测试：Blob 编码/解码
# ─────────────────────────────────────────────
class TestBlobEncoding:
    """测试 blob 编码/解码"""

    def test_encode_decode_round_trip(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        original = b"hello world"
        encoded = BaseRedisSaver._encode_blob("msgpack", original)
        assert encoded.startswith("msgpack:")
        type_tag, decoded = BaseRedisSaver._decode_blob(encoded)
        assert type_tag == "msgpack"
        assert decoded == original

    def test_encode_none(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        encoded = BaseRedisSaver._encode_blob("msgpack", None)
        assert encoded == "empty:"
        type_tag, decoded = BaseRedisSaver._decode_blob(encoded)
        assert type_tag == "empty"
        assert decoded is None


# ─────────────────────────────────────────────
# 测试：版本管理
# ─────────────────────────────────────────────
class TestVersioning:
    """测试版本号生成"""

    def test_first_version(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        saver = BaseRedisSaver.__new__(BaseRedisSaver)
        v = saver.get_next_version(None, None)
        assert v.startswith("00000000000000000000000000000001.")

    def test_increment_version(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        saver = BaseRedisSaver.__new__(BaseRedisSaver)
        v1 = saver.get_next_version(None, None)
        v2 = saver.get_next_version(v1, None)
        assert v2.startswith("00000000000000000000000000000002.")

    def test_int_version(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        saver = BaseRedisSaver.__new__(BaseRedisSaver)
        v = saver.get_next_version(5, None)
        assert v.startswith("00000000000000000000000000000006.")


# ─────────────────────────────────────────────
# 测试：metadata 过滤
# ─────────────────────────────────────────────
class TestMetadataFilter:
    """测试 metadata 过滤逻辑"""

    def test_exact_match(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        metadata = {"source": "loop", "step": 1}
        assert BaseRedisSaver._metadata_matches(metadata, {"source": "loop"}) is True
        assert BaseRedisSaver._metadata_matches(metadata, {"source": "input"}) is False

    def test_nested_match(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        metadata = {"parents": {"thread-1": "cp-1"}}
        assert BaseRedisSaver._metadata_matches(
            metadata, {"parents": {"thread-1": "cp-1"}}
        ) is True

    def test_missing_key(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        metadata = {"source": "loop"}
        assert BaseRedisSaver._metadata_matches(metadata, {"step": 1}) is False

    def test_empty_filter(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        metadata = {"source": "loop"}
        assert BaseRedisSaver._metadata_matches(metadata, {}) is True


# ─────────────────────────────────────────────
# 测试：Blob dump/load（BaseRedisSaver 方法）
# ─────────────────────────────────────────────
class TestBlobDumpLoad:
    """测试 blob dump/load 逻辑"""

    @pytest.fixture
    def saver(self):
        """创建一个 BaseRedisSaver 实例用于测试"""
        from app.core.redis_checkpointer import BaseRedisSaver

        s = BaseRedisSaver.__new__(BaseRedisSaver)
        s.serde = _serde
        return s

    def test_dump_blobs_with_values(self, saver):
        """dump_blobs 正确序列化有值的 channel"""
        messages = [HumanMessage(content="hello")]
        fields = saver._dump_blobs({"messages": messages}, {"messages": "1"})
        assert "b:messages:1" in fields
        assert fields["b:messages:1"].startswith("msgpack:")

    def test_dump_blobs_empty(self, saver):
        """dump_blobs 对缺失的 channel 生成 empty 标记"""
        fields = saver._dump_blobs({}, {"messages": "1"})
        assert fields["b:messages:1"] == "empty:"

    def test_dump_blobs_no_versions(self, saver):
        """空 versions 返回空字典"""
        fields = saver._dump_blobs({}, {})
        assert fields == {}

    def test_load_blobs_from_hash(self, saver):
        """从 Hash 数据加载 blobs"""
        messages = [HumanMessage(content="hello")]
        type_tag, raw_bytes = _serde.dumps_typed(messages)
        encoded = f"{type_tag}:{base64.b64encode(raw_bytes).decode('utf-8')}"

        hash_data = {"b:messages:1": encoded}
        result = saver._load_blobs(hash_data, {"messages": "1"}, set())
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)

    def test_load_blobs_skip_inline(self, saver):
        """跳过已内联的 channel"""
        hash_data = {"b:messages:1": "some_data"}
        result = saver._load_blobs(hash_data, {"messages": "1"}, {"messages"})
        assert result == {}

    def test_load_blobs_skip_empty(self, saver):
        """跳过 empty 标记的 blobs"""
        result = saver._load_blobs({"b:messages:1": "empty:"}, {"messages": "1"}, set())
        assert result == {}


# ─────────────────────────────────────────────
# 测试：Writes dump/load
# ─────────────────────────────────────────────
class TestWritesDumpLoad:
    """测试 writes dump/load 逻辑"""

    @pytest.fixture
    def saver(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        s = BaseRedisSaver.__new__(BaseRedisSaver)
        s.serde = _serde
        return s

    def test_dump_writes(self, saver):
        """dump_writes 正确序列化写入"""
        writes = [("messages", HumanMessage(content="hello"))]
        fields = saver._dump_writes("cp-001", "task-1", "", writes)
        assert len(fields) == 1
        # field 格式: w:{checkpoint_id}:{task_id}:{idx}
        field_key = list(fields.keys())[0]
        assert field_key.startswith("w:cp-001:task-1:")

        # value 是 JSON
        data = json.loads(fields[field_key])
        assert data["channel"] == "messages"
        assert data["type"] == "msgpack"
        assert "blob" in data

    def test_load_writes(self, saver):
        """load_writes 正确反序列化写入"""
        # 先 dump
        writes = [("messages", HumanMessage(content="hello"))]
        fields = saver._dump_writes("cp-001", "task-1", "", writes)

        # 再 load
        result = saver._load_writes(fields, "cp-001")
        assert len(result) == 1
        task_id, channel, value = result[0]
        assert task_id == "task-1"
        assert channel == "messages"
        assert isinstance(value, HumanMessage)
        assert value.content == "hello"

    def test_dump_writes_empty(self, saver):
        """空 writes 返回空字典"""
        fields = saver._dump_writes("cp-001", "task-1", "", [])
        assert fields == {}

    def test_load_writes_empty(self, saver):
        """空 writes 返回空列表"""
        result = saver._load_writes({}, "cp-001")
        assert result == []


# ─────────────────────────────────────────────
# 测试：Order 管理
# ─────────────────────────────────────────────
class TestOrderManagement:
    """测试 checkpoint 顺序管理"""

    def test_update_order_new(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        order = BaseRedisSaver._update_order([], "cp-001")
        assert order == ["cp-001"]

    def test_update_order_append(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        order = BaseRedisSaver._update_order(["cp-001"], "cp-002")
        assert order == ["cp-002", "cp-001"]

    def test_update_order_dedup(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        order = BaseRedisSaver._update_order(["cp-001", "cp-002"], "cp-001")
        assert order == ["cp-001", "cp-002"]

    def test_parse_order(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        hash_data = {"__order__": '["cp-002", "cp-001"]'}
        order = BaseRedisSaver._parse_order(hash_data)
        assert order == ["cp-002", "cp-001"]

    def test_parse_order_empty(self):
        from app.core.redis_checkpointer import BaseRedisSaver

        order = BaseRedisSaver._parse_order({})
        assert order == []


# ─────────────────────────────────────────────
# 测试：AsyncRedisSaver 完整流程（使用 mock）
# ─────────────────────────────────────────────
class TestAsyncRedisSaverUnit:
    """AsyncRedisSaver 单元测试（使用 mock）"""

    @pytest.fixture
    def mock_redis(self):
        """创建 mock Redis 客户端"""
        redis = AsyncMock()
        redis.hgetall = AsyncMock(return_value={})
        redis.hget = AsyncMock(return_value=None)
        redis.hexists = AsyncMock(return_value=False)
        redis.hset = AsyncMock()  # 新实现中 hset 是 await 调用
        redis.expire = AsyncMock()  # 新实现中 expire 是 await 调用
        redis.delete = AsyncMock()
        redis.ping = AsyncMock()

        async def empty_scan(*args, **kwargs):
            return
            yield

        redis.scan_iter = MagicMock(return_value=empty_scan())

        return redis

    @pytest.fixture
    def checkpointer(self, mock_redis):
        """创建 AsyncRedisSaver 实例"""
        from app.core.redis_checkpointer import AsyncRedisSaver

        cp = AsyncRedisSaver(mock_redis, ttl=3600)
        return cp

    @pytest.mark.asyncio
    async def test_aput_stores_to_single_hash(self, checkpointer, mock_redis):
        """aput 将所有数据存储到单个 Hash"""
        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "",
            }
        }
        checkpoint = create_test_checkpoint()
        metadata = create_test_metadata()

        result = await checkpointer.aput(
            config, checkpoint, metadata, {"messages": 1}
        )

        assert result["configurable"]["checkpoint_id"] == "test-cp-001"

        # 验证 hset 被调用
        assert mock_redis.hset.called
        call_args = mock_redis.hset.call_args
        key = call_args[0][0]
        assert key == "ckp:test-thread:"

        # 验证 mapping 包含 checkpoint 结构
        mapping = call_args[1]["mapping"]
        assert "c:test-cp-001" in mapping
        assert "t:test-cp-001" in mapping
        assert "m:test-cp-001" in mapping
        assert "__order__" in mapping

    @pytest.mark.asyncio
    async def test_aput_inline_primitives(self, checkpointer, mock_redis):
        """aput 将原始类型内联到 checkpoint 中，复杂类型存入 blob"""
        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "",
            }
        }
        checkpoint = create_test_checkpoint()
        metadata = create_test_metadata()

        await checkpointer.aput(config, checkpoint, metadata, {"messages": 1})

        # 验证 blob field 存在（messages 是 list，非原始类型）
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]
        assert "b:messages:1" in mapping

    @pytest.mark.asyncio
    async def test_aput_writes(self, checkpointer, mock_redis):
        """aput_writes 正确存储中间写入"""
        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-001",
            }
        }
        writes = [("messages", HumanMessage(content="hello"))]

        await checkpointer.aput_writes(config, writes, "task-1")

        # 验证 hset 被调用
        assert mock_redis.hset.called
        call_args = mock_redis.hset.call_args
        key = call_args[0][0]
        assert key == "ckp:test-thread:"

        mapping = call_args[1]["mapping"]
        # 验证 write field 存在
        assert any(k.startswith("w:cp-001:task-1:") for k in mapping.keys())

    @pytest.mark.asyncio
    async def test_aget_tuple_empty(self, checkpointer, mock_redis):
        """空 checkpoint 返回 None"""
        config = {
            "configurable": {
                "thread_id": "non-existent",
                "checkpoint_ns": "",
            }
        }
        mock_redis.hgetall.return_value = {}
        result = await checkpointer.aget_tuple(config)
        assert result is None

    @pytest.mark.asyncio
    async def test_adelete_thread(self, checkpointer, mock_redis):
        """adelete_thread 删除所有相关 key"""
        async def async_keys():
            for k in ["ckp:test-thread:"]:
                yield k

        mock_redis.scan_iter = MagicMock(return_value=async_keys())

        await checkpointer.adelete_thread("test-thread")

        assert mock_redis.delete.called

    @pytest.mark.asyncio
    async def test_setup(self, checkpointer, mock_redis):
        """setup 验证连接"""
        await checkpointer.setup()
        assert mock_redis.ping.called

    @pytest.mark.asyncio
    async def test_ttl_applied(self, mock_redis):
        """TTL 被正确应用到 Hash key"""
        from app.core.redis_checkpointer import AsyncRedisSaver

        cp = AsyncRedisSaver(mock_redis, ttl=3600)
        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "",
            }
        }
        checkpoint = create_test_checkpoint()
        metadata = create_test_metadata()

        await cp.aput(config, checkpoint, metadata, {"messages": 1})

        # 验证 expire 被调用
        assert mock_redis.expire.called
        call_args = mock_redis.expire.call_args
        key = call_args[0][0]
        ttl = call_args[0][1]
        assert key == "ckp:test-thread:"
        assert ttl == 3600

    @pytest.mark.asyncio
    async def test_backward_compat_alias(self):
        """RedisCheckpointer 别名指向 AsyncRedisSaver"""
        from app.core.redis_checkpointer import AsyncRedisSaver, RedisMemorySaver

        assert RedisMemorySaver is AsyncRedisSaver


# ─────────────────────────────────────────────
# 测试：RedisSaver 同步实现（使用 mock）
# ─────────────────────────────────────────────
class TestRedisSaverUnit:
    """RedisSaver 同步实现单元测试"""

    @pytest.fixture
    def mock_redis_sync(self):
        """创建同步 mock Redis 客户端"""
        redis = MagicMock()
        redis.hgetall = MagicMock(return_value={})
        redis.hget = MagicMock(return_value=None)
        redis.hexists = MagicMock(return_value=False)
        redis.hset = MagicMock()
        redis.expire = MagicMock()
        redis.delete = MagicMock()
        redis.ping = MagicMock()
        redis.scan = MagicMock(return_value=(0, []))

        return redis

    @pytest.fixture
    def checkpointer(self, mock_redis_sync):
        """创建 RedisSaver 实例"""
        from app.core.redis_checkpointer import RedisSaver

        cp = RedisSaver(mock_redis_sync, ttl=3600)
        return cp

    def test_put_stores_to_single_hash(self, checkpointer, mock_redis_sync):
        """put 正确存储 checkpoint"""
        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "",
            }
        }
        checkpoint = create_test_checkpoint()
        metadata = create_test_metadata()

        result = checkpointer.put(config, checkpoint, metadata, {"messages": 1})

        assert result["configurable"]["checkpoint_id"] == "test-cp-001"
        assert mock_redis_sync.hset.called
        call_args = mock_redis_sync.hset.call_args
        key = call_args[0][0]
        assert key == "ckp:test-thread:"

    def test_get_tuple_empty(self, checkpointer, mock_redis_sync):
        """空 checkpoint 返回 None"""
        config = {
            "configurable": {
                "thread_id": "non-existent",
                "checkpoint_ns": "",
            }
        }
        mock_redis_sync.hgetall.return_value = {}
        result = checkpointer.get_tuple(config)
        assert result is None

    def test_delete_thread(self, checkpointer, mock_redis_sync):
        """delete_thread 扫描并删除所有相关 key"""
        mock_redis_sync.scan.side_effect = [
            (0, ["ckp:test-thread:"]),
        ]

        checkpointer.delete_thread("test-thread")

        assert mock_redis_sync.delete.called

    def test_setup(self, checkpointer, mock_redis_sync):
        """setup 验证连接"""
        checkpointer.setup()
        assert mock_redis_sync.ping.called

    def test_put_writes(self, checkpointer, mock_redis_sync):
        """put_writes 正确存储中间写入"""
        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-001",
            }
        }
        writes = [("messages", HumanMessage(content="hello"))]

        checkpointer.put_writes(config, writes, "task-1")

        assert mock_redis_sync.hset.called
        call_args = mock_redis_sync.hset.call_args
        key = call_args[0][0]
        assert key == "ckp:test-thread:"


# ─────────────────────────────────────────────
# 测试：单 Hash 模型验证
# ─────────────────────────────────────────────
class TestSingleHashModel:
    """验证单 Hash 合并模型"""

    @pytest.fixture
    def mock_redis(self):
        redis = AsyncMock()
        redis.hgetall = AsyncMock(return_value={})
        redis.hget = AsyncMock(return_value=None)
        redis.hexists = AsyncMock(return_value=False)
        redis.hset = AsyncMock()
        redis.expire = AsyncMock()
        redis.ping = AsyncMock()
        return redis

    @pytest.mark.asyncio
    async def test_single_key_per_thread(self, mock_redis):
        """每个 thread 只使用 1 个 Redis key"""
        from app.core.redis_checkpointer import AsyncRedisSaver

        cp = AsyncRedisSaver(mock_redis, ttl=3600)
        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "",
            }
        }

        # 模拟 10 轮对话
        for i in range(10):
            checkpoint = create_test_checkpoint(f"cp-{i:03d}")
            metadata = create_test_metadata()
            await cp.aput(config, checkpoint, metadata, {"messages": i + 1})

        # 验证所有操作都使用同一个 key
        hset_calls = mock_redis.hset.call_args_list
        keys_used = set()
        for call in hset_calls:
            key = call[0][0]
            keys_used.add(key)

        assert len(keys_used) == 1
        assert "ckp:test-thread:" in keys_used

    @pytest.mark.asyncio
    async def test_ttl_set_once(self, mock_redis):
        """TTL 只设置 1 次（而非每个 checkpoint 一次）"""
        from app.core.redis_checkpointer import AsyncRedisSaver

        cp = AsyncRedisSaver(mock_redis, ttl=3600)
        config = {
            "configurable": {
                "thread_id": "test-thread",
                "checkpoint_ns": "",
            }
        }

        # 模拟 5 轮对话
        for i in range(5):
            checkpoint = create_test_checkpoint(f"cp-{i:03d}")
            metadata = create_test_metadata()
            await cp.aput(config, checkpoint, metadata, {"messages": i + 1})

        # 验证 expire 调用次数
        expire_calls = mock_redis.expire.call_args_list
        # 每次 aput 都会设置一次 TTL（因为每次都要更新 key）
        # 但 key 是同一个，所以 TTL 会刷新
        assert len(expire_calls) == 5  # 5 次 aput，每次 1 次 expire


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
