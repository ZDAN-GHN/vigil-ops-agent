"""消息队列服务

基于 Redis List 实现的轻量级消息队列，用于异步持久化对话历史到 MySQL。

架构设计：
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  生产者      │────▶│ Redis List   │────▶│ 消费者协程   │
│ (rag_agent) │     │ (持久化队列)  │     │ (后台运行)   │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     │
       │ 失败               │ 不可用              │ 失败
       ▼                    ▼                     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ asyncio.Queue│     │ 降级到内存队列 │     │ 重试 3 次    │
│ (内存兜底)   │     │              │     │ 记录失败日志  │
└─────────────┘     └──────────────┘     └─────────────┘

核心功能：
1. enqueue() - 生产者入队消息（优先 Redis，失败降级到内存）
2. start_consumer() - 启动后台消费者协程
3. stop_consumer() - 优雅停止消费者
4. 批量消费 + 重试机制
5. 失败消息记录到兜底日志文件

使用示例：
    # 启动时
    await message_queue_service.start_consumer()
    
    # 生产中
    await message_queue_service.enqueue("session-123", messages)
    
    # 关闭时
    await message_queue_service.stop_consumer()
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from app.config import config


class MessageQueueService:
    """消息队列服务
    
    提供 Redis List 消息队列 + 内存队列兜底 + 后台消费者。
    """
    
    def __init__(self) -> None:
        """初始化消息队列服务"""
        self._redis_client = None
        self._memory_queue: asyncio.Queue = asyncio.Queue()
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False
        self._queue_key = config.mq_queue_key
        self._batch_size = config.mq_batch_size
        self._retry_count = config.mq_retry_count
        self._fallback_log_path = Path(config.mq_fallback_log_file)
        self._use_memory_fallback = False  # 是否使用内存兜底
        
        # 确保兜底日志目录存在
        self._fallback_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def _get_redis_client(self):
        """获取 Redis 客户端（延迟初始化）"""
        if self._redis_client is None:
            from app.core.redis_client import redis_manager
            self._redis_client = await redis_manager.get_client()
        return self._redis_client
    
    async def enqueue(
        self,
        session_id: str,
        messages: list[Any],
        start_order: int = 0,
    ) -> bool:
        """将消息入队
        
        优先使用 Redis List，失败时降级到内存队列。
        
        Args:
            session_id: 会话 ID
            messages: 消息列表（已序列化的字典）
            start_order: 起始序号
            
        Returns:
            bool: 是否成功入队
        """
        if not messages:
            return False
        
        # 构造消息体
        payload = {
            "session_id": session_id,
            "messages": messages,
            "start_order": start_order,
            "timestamp": datetime.now().isoformat(),
        }
        message_json = json.dumps(payload, ensure_ascii=False, default=str)
        
        # 尝试推入 Redis
        try:
            redis_client = await self._get_redis_client()
            await redis_client.lpush(self._queue_key, message_json)
            logger.debug(
                f"消息已入队 Redis: session={session_id}, "
                f"count={len(messages)}, queue_key={self._queue_key}"
            )
            self._use_memory_fallback = False  # 重置降级标志
            return True
        except Exception as e:
            logger.warning(
                f"Redis 入队失败，降级到内存队列: {e}"
            )
            self._use_memory_fallback = True
        
        # 降级到内存队列
        try:
            self._memory_queue.put_nowait(message_json)
            logger.info(
                f"消息已入队内存队列: session={session_id}, "
                f"count={len(messages)}"
            )
            return True
        except Exception as e:
            logger.error(f"内存队列入队失败: {e}")
            # 记录到兜底日志
            self._log_fallback_message(message_json, str(e))
            return False
    
    async def start_consumer(self) -> None:
        """启动后台消费者协程"""
        if self._running:
            logger.warning("消费者已在运行")
            return
        
        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_loop())
        logger.info(
            f"消息队列消费者已启动: queue_key={self._queue_key}, "
            f"batch_size={self._batch_size}"
        )
    
    async def stop_consumer(self) -> None:
        """优雅停止消费者
        
        等待当前批次处理完成，并处理剩余消息。
        """
        if not self._running:
            return
        
        logger.info("正在停止消息队列消费者...")
        self._running = False
        
        if self._consumer_task:
            # 等待当前任务完成
            try:
                await asyncio.wait_for(self._consumer_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("消费者停止超时，强制取消")
                self._consumer_task.cancel()
            
            self._consumer_task = None
        
        # 处理剩余消息
        await self._drain_remaining_messages()
        
        logger.info("消息队列消费者已停止")
    
    async def _consume_loop(self) -> None:
        """消费者主循环"""
        logger.info("消费者循环开始运行")
        
        while self._running:
            try:
                # 处理一批消息
                processed = await self._process_batch()
                
                if processed == 0:
                    # 无消息，等待一段时间
                    await asyncio.sleep(1.0)
                else:
                    logger.debug(f"本批次处理 {processed} 条消息")
            
            except Exception as e:
                logger.error(f"消费者循环异常: {e}")
                await asyncio.sleep(2.0)  # 异常后等待一段时间
        
        logger.info("消费者循环结束")
    
    async def _process_batch(self) -> int:
        """处理一批消息
        
        Returns:
            int: 成功处理的消息数量
        """
        processed = 0
        
        # 优先从 Redis 消费
        if not self._use_memory_fallback:
            try:
                processed = await self._process_redis_batch()
                if processed > 0:
                    return processed
            except Exception as e:
                logger.warning(f"Redis 消费失败，切换到内存队列: {e}")
                self._use_memory_fallback = True
        
        # 从内存队列消费
        processed = await self._process_memory_batch()
        return processed
    
    async def _process_redis_batch(self) -> int:
        """从 Redis 消费一批消息"""
        redis_client = await self._get_redis_client()
        
        # 批量获取消息
        pipe = redis_client.pipeline()
        for _ in range(self._batch_size):
            pipe.rpop(self._queue_key)
        results = await pipe.execute()
        
        # 过滤空值
        messages = [r for r in results if r is not None]
        if not messages:
            return 0
        
        # 处理每条消息
        success_count = 0
        failed_messages = []
        
        for msg_json in messages:
            try:
                payload = json.loads(msg_json)
                await self._persist_to_mysql(payload)
                success_count += 1
            except Exception as e:
                logger.error(f"消息持久化失败: {e}")
                failed_messages.append(msg_json)
        
        # 失败消息重试
        for msg_json in failed_messages:
            await self._retry_failed_message(msg_json)
        
        return success_count
    
    async def _process_memory_batch(self) -> int:
        """从内存队列消费一批消息"""
        processed = 0
        
        while processed < self._batch_size and not self._memory_queue.empty():
            try:
                msg_json = self._memory_queue.get_nowait()
                payload = json.loads(msg_json)
                await self._persist_to_mysql(payload)
                processed += 1
                self._memory_queue.task_done()
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"内存队列消息处理失败: {e}")
                self._log_fallback_message(msg_json, str(e))
                self._memory_queue.task_done()
        
        return processed
    
    async def _persist_to_mysql(self, payload: dict) -> None:
        """将消息持久化到 MySQL
        
        Args:
            payload: 消息体
        """
        from app.services.conversation_history_service import (
            conversation_history_service,
        )
        
        session_id = payload["session_id"]
        messages = payload["messages"]
        start_order = payload.get("start_order", 0)
        
        # 反序列化消息
        from app.services.conversation_history_service import (
            ConversationHistoryService,
        )
        
        deserialized_messages = []
        for msg_dict in messages:
            msg = ConversationHistoryService._reconstruct_from_dict(msg_dict)
            if msg:
                deserialized_messages.append(msg)
        
        if not deserialized_messages:
            logger.warning(f"无可持久化的消息: session={session_id}")
            return
        
        # 写入 MySQL
        count = await conversation_history_service.save_messages(
            session_id,
            deserialized_messages,
            start_order=start_order,
        )
        
        if count > 0:
            logger.info(
                f"消息已持久化到 MySQL: session={session_id}, count={count}"
            )
    
    async def _retry_failed_message(self, msg_json: str) -> None:
        """重试失败的消息
        
        Args:
            msg_json: 消息 JSON
        """
        for attempt in range(1, self._retry_count + 1):
            try:
                payload = json.loads(msg_json)
                await self._persist_to_mysql(payload)
                logger.info(f"消息重试成功 (attempt {attempt})")
                return
            except Exception as e:
                logger.warning(
                    f"消息重试失败 (attempt {attempt}/{self._retry_count}): {e}"
                )
                if attempt < self._retry_count:
                    await asyncio.sleep(1.0)  # 重试间隔
        
        # 所有重试失败，记录到兜底日志
        logger.error(f"消息重试全部失败，记录到兜底日志")
        self._log_fallback_message(msg_json, "所有重试失败")
    
    def _log_fallback_message(self, msg_json: str, reason: str) -> None:
        """记录失败消息到兜底日志文件
        
        Args:
            msg_json: 消息 JSON
            reason: 失败原因
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "message": msg_json,
            }
            with open(self._fallback_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            logger.warning(f"失败消息已记录到兜底日志: {self._fallback_log_path}")
        except Exception as e:
            logger.error(f"记录兜底日志失败: {e}")
    
    async def _drain_remaining_messages(self) -> None:
        """处理剩余消息（关闭前调用）"""
        logger.info("处理剩余消息...")
        
        # 处理 Redis 中的剩余消息
        try:
            while True:
                processed = await self._process_redis_batch()
                if processed == 0:
                    break
        except Exception as e:
            logger.error(f"处理 Redis 剩余消息失败: {e}")
        
        # 处理内存队列中的剩余消息
        try:
            while not self._memory_queue.empty():
                await self._process_memory_batch()
        except Exception as e:
            logger.error(f"处理内存队列剩余消息失败: {e}")
        
        logger.info("剩余消息处理完成")
    
    async def get_queue_size(self) -> dict:
        """获取队列大小
        
        Returns:
            dict: {"redis": int, "memory": int}
        """
        redis_size = 0
        memory_size = self._memory_queue.qsize()
        
        if not self._use_memory_fallback:
            try:
                redis_client = await self._get_redis_client()
                redis_size = await redis_client.llen(self._queue_key)
            except Exception:
                pass
        
        return {
            "redis": redis_size,
            "memory": memory_size,
            "total": redis_size + memory_size,
        }


# 全局单例
message_queue_service = MessageQueueService()
