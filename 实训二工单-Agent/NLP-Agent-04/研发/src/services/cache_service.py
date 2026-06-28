"""缓存服务 — 支持 Memory 和 Redis 两种后端"""

from __future__ import annotations

import json
import time
from threading import Lock
from typing import Any, Optional

from config import settings
from src.core.models import AnswerResult


class CacheService:
    """问答结果缓存"""

    def __init__(self, cache_type: Optional[str] = None):
        self._type = cache_type or settings.CACHE_TYPE
        self._memory: dict[str, tuple[float, str]] = {}
        self._lock = Lock()
        self._ttl = 300  # 5 分钟
        self._redis = None

        if self._type == "redis":
            self._init_redis()

    def _init_redis(self):
        try:
            import redis
            self._redis = redis.Redis.from_url(settings.REDIS_URL)
            self._redis.ping()
        except Exception:
            self._type = "memory"  # Redis 不可用则回退内存

    def get(self, key: str) -> Optional[AnswerResult]:
        """获取缓存"""
        if self._type == "redis":
            return self._get_redis(key)
        return self._get_memory(key)

    def set(self, key: str, value: AnswerResult) -> None:
        """设置缓存（仅缓存成功的查询结果）"""
        if not value.success:
            return
        if self._type == "redis":
            self._set_redis(key, value)
        else:
            self._set_memory(key, value)

    def _get_memory(self, key: str) -> Optional[AnswerResult]:
        with self._lock:
            item = self._memory.get(key)
            if item is None:
                return None
            timestamp, data_json = item
            if time.time() - timestamp > self._ttl:
                del self._memory[key]
                return None
            return AnswerResult(**json.loads(data_json))

    def _set_memory(self, key: str, value: AnswerResult) -> None:
        with self._lock:
            data_json = json.dumps({
                "question": value.question,
                "answer": value.answer,
                "sql": value.sql,
                "tables_used": value.tables_used,
                "model_used": value.model_used,
                "success": value.success,
            }, ensure_ascii=False)
            self._memory[key] = (time.time(), data_json)

    def _get_redis(self, key: str) -> Optional[AnswerResult]:
        try:
            data = self._redis.get(f"nl2sql:{key}")
            if data:
                return AnswerResult(**json.loads(data))
        except Exception:
            pass
        return None

    def _set_redis(self, key: str, value: AnswerResult) -> None:
        try:
            self._redis.setex(
                f"nl2sql:{key}",
                self._ttl,
                json.dumps({
                    "question": value.question,
                    "answer": value.answer,
                    "sql": value.sql,
                    "tables_used": value.tables_used,
                    "model_used": value.model_used,
                    "success": value.success,
                }, ensure_ascii=False),
            )
        except Exception:
            pass

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()
