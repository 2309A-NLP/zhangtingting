import json
from typing import Any
# 使用 Redis 列表（List）作为任务队列，
# 实现生产者-消费者模式：扫描器将任务 ID 入队（RPUSH），Worker 从队列取任务（LPOP）。
from redis.asyncio import Redis
from structlog import get_logger

from app.core.config import settings

logger = get_logger()

_redis_client: Redis | None = None


class ReminderQueueBackend:
    def __init__(self, client: Redis | None = None) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def enqueue_task(self, task_id: int) -> None:
        if self._client is None:
            return
        try:
            # 操作	                                说明
            # rpush	                               从右侧推入（列表尾部） → FIFO 队列
            # json.dumps({"task_id": task_id})	   序列化任务数据（可扩展）
            # hasattr(result, "__await__")	       兼容异步/同步 Redis 客户端
            # 什么是 __await__？
            # Python 协程协议
            # 条件	                           说明	                           行为
            # hasattr(result, "__await__")	   检查对象是否有 __await__ 方法	 有 = 是协程（Coroutine）
            # True	                           是协程	                     await result → 获取结果
            # False	                           不是协程	                     直接使用 result
            result: Any = self._client.rpush(settings.redis_queue_key, json.dumps({"task_id": task_id}))
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:  # pragma: no cover
            logger.warning("redis_queue_enqueue_failed", task_id=task_id, error=str(exc))

    async def dequeue_task_ids(self, *, max_items: int) -> list[int]:
        if self._client is None:
            return []
        try:
            task_ids: list[int] = []
            for _ in range(max_items):
                result: Any = self._client.lpop(settings.redis_queue_key)
                payload = await result if hasattr(result, "__await__") else result
                if payload is None:
                    break
                if not isinstance(payload, str):
                    continue
                # 解析 JSON 并提取 task_id
                data = json.loads(payload)
                task_id = data.get("task_id")
                # 验证并添加到结果列表
                if isinstance(task_id, int):
                    task_ids.append(task_id)
            return task_ids
        except Exception as exc:  # pragma: no cover
            logger.warning("redis_queue_dequeue_failed", error=str(exc))
            return []

    # 获取队列长度
    async def get_queue_length(self) -> int:
        if self._client is None:
            return 0
        try:
            result: Any = self._client.llen(settings.redis_queue_key)
            length = await result if hasattr(result, "__await__") else result
            return int(length)
        except Exception as exc:  # pragma: no cover
            logger.warning("redis_queue_length_failed", error=str(exc))
            return 0


def get_redis_client() -> Redis | None:
    global _redis_client
    if not settings.redis_enabled:
        return None
    if _redis_client is None:
        # from_url 是 Redis 客户端库提供的一个类方法（classmethod），用于从 URL 字符串快速创建 Redis 客户端实例。
        _redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        logger.info("redis_queue_client_initialized", redis_url=settings.redis_url)
    return _redis_client


def get_reminder_queue_backend() -> ReminderQueueBackend:
    return ReminderQueueBackend(get_redis_client())
