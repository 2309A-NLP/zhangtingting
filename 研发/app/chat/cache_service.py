from __future__ import annotations
'''
这是一个缓存服务，用于缓存 LLM 的完整响应，避免重复调用相同的请求。
'''
import json
# ValidationError：Pydantic 的验证错误，当从字典创建 ChatResponse 对象失败时抛出
from pydantic import ValidationError

from app.api.schemas import ChatResponse
from app.core.config import get_settings
from app.core.logging import get_logger
# query_cache_key：生成缓存键的函数
from app.db.redis_client import get_redis, query_cache_key

logger = get_logger(__name__)


class ChatCacheService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.redis = get_redis()

    async def get_cached_response(
        self,
        *,
        user_id: str,
        role_id: str,
        session_id: str | None = None,
        query: str,
    ) -> ChatResponse | None:
        '''
        返回值：
            如果缓存命中：返回 ChatResponse 对象
            如果缓存未命中：返回 None
        '''
        # 生成唯一键。这个函数会把 query 和 session_id 等信息组合成一个 Redis key。
        # 哈希检索 只有问题完全一样（一字不差）才能命中。
        key = query_cache_key(user_id, role_id, query, session_id)
        cached_payload = await self.redis.get(key)
        if not cached_payload:
            logger.info("chat_cache_miss", user_id=user_id, role_id=role_id)
            return None

        try:
            payload = json.loads(cached_payload)  # 转字典
            # 用 Pydantic 验证并创建 ChatResponse 对象
            # model_validate 是 Pydantic 模型的一个<类方法>，用于从字典创建对象。
            # 还没有对象，需要用这个方法去创建对象  所以是类方法才行
            cached_response = ChatResponse.model_validate(payload)
            '''
            存的时候：
            ChatResponse 对象 → model_dump() → 字典 → json.dumps() → JSON 字符串 → Redis
            读的时候：
            Redis 字符串 → json.loads() → 字典 → model_validate() → ChatResponse 对象
            
            问题	答案
            model_validate 做什么？	         把字典转成 Pydantic 对象
            为什么需要它？	                 自动验证类型、自动转换数据
            和 model_dump 什么关系？	         互为逆操作
            为什么不用 **payload？	         不会自动转换类型，容易出错
            一句话：model_validate 是 Pydantic 提供的"安全反序列化"方法，保证从缓存读出来的数据是正确格式。
            '''
        except (json.JSONDecodeError, ValidationError) as exc:
            # 聊天缓存反序列化失败
            logger.warning("chat_cache_deserialize_failed", user_id=user_id, role_id=role_id, error=str(exc))
            # 如果任一失败：删掉坏的缓存记录，返回 None
            await self.redis.delete(key)
            return None

        logger.info("chat_cache_hit", user_id=user_id, role_id=role_id)
        return cached_response

    # LLM 调用成功后，把响应存入缓存。
    async def set_cached_response(
        self,
        *,
        user_id: str,
        role_id: str,
        session_id: str | None = None,
        query: str,
        response: ChatResponse,
    ) -> None:
        key = query_cache_key(user_id, role_id, query, session_id)
        # response.model_dump(mode="json")：把 ChatResponse 对象转成字典
        # mode="json" 的作用：确保数据类型是 JSON 兼容的。比如把 datetime 转成 ISO 字符串，把 Decimal 转成浮点数。
        # json.dumps(..., ensure_ascii=False)：把字典转成 JSON 字符串，保留中文
        payload = json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
        # 存字符串
        await self.redis.set(key, payload, ex=self.settings.redis_query_cache_ttl_seconds)
        logger.info("chat_cache_set", user_id=user_id, role_id=role_id)
