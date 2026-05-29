"""
Redis客户端管理模块

提供Redis连接管理、键命名规范和常用键构建函数
所有Redis键都采用命名空间隔离，格式为: prefix:user_id:role_id:suffix
"""

from collections.abc import AsyncGenerator
from hashlib import sha256
from urllib.parse import quote

from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Redis客户端单例
_redis_client: Redis | None = None
'''
优点
特点	               说明
单例模式	整个应用共用一个Redis连接，不浪费资源
延迟初始化	连接在启动时创建，不是每次使用时创建
全局可访问	任何地方都能方便地获取
防止重复连接	不会创建多个连接池
只能通过这个函数获取
'''


def _normalize_key_segment(value: str) -> str:
    """
    规范化Redis键的片段
    Args:
        value: 待规范化的字符串
    Returns:
        URL编码后的字符串
    Raises:
        ValueError: 如果输入为空
    Note:
        使用URL编码确保键中不包含特殊字符
        保留连字符和下划线作为安全字符
    """
    normalized = value.strip()
    if not normalized:
        raise ValueError("Redis key segment cannot be empty.")
    '''
    作用：对字符串进行 URL 编码（百分号编码），确保特殊字符（如 ? & = # 空格 中文 等）被转成 %XX 形式，避免破坏 Redis key 的结构或引起歧义。
    safe="-_" 参数表示：允许保留连字符 - 和下划线 _ 不编码，因为它们通常是安全的、可读的分隔符或标识符。
    URL 编码 = 把不安全的特殊字符转成 %XX 形式，让它们能安全地在 URL 或类似场景中传输和存储。
    '''
    return quote(normalized, safe="-_")


def build_namespaced_key(prefix: str, user_id: str, role_id: str, suffix: str) -> str:
    """
    构建带命名空间的Redis键

    Args:
        prefix: 键前缀，标识数据类型
        user_id: 用户ID
        role_id: 角色ID
        suffix: 键后缀

    Returns:
        格式化的Redis键，格式为: prefix:user_id:role_id:suffix

    Example:
        >>> build_namespaced_key("chat", "user123", "role01", "recent")
        "chat:user123:role01:recent"
    """
    return (
        f"{_normalize_key_segment(prefix)}:"
        f"{_normalize_key_segment(user_id)}:"
        f"{_normalize_key_segment(role_id)}:"
        f"{_normalize_key_segment(suffix)}"
    )


def _session_suffix(base: str, session_id: str | None = None) -> str:
    """
    构建会话后缀

    Args:
        base: 基础后缀
        session_id: 会话ID，可选

    Returns:
        如果有session_id，返回"base:session_id"，否则返回"base"
    """
    return base if not session_id else f"{base}:{session_id}"


def chat_recent_key(user_id: str, role_id: str, session_id: str | None = None) -> str:
    """
    构建聊天最近消息的Redis键

    Args:
        user_id: 用户ID
        role_id: 角色ID
        session_id: 会话ID，可选

    Returns:
        Redis键字符串
    """
    return build_namespaced_key("chat", user_id, role_id, _session_suffix("recent", session_id))


def chat_session_key(user_id: str, role_id: str, session_id: str | None = None) -> str:
    """
    构建聊天会话的Redis键

    Args:
        user_id: 用户ID
        role_id: 角色ID
        session_id: 会话ID，可选

    Returns:
        Redis键字符串
    """
    return build_namespaced_key("chat", user_id, role_id, _session_suffix("session", session_id))


def rate_limit_key(user_id: str, role_id: str) -> str:
    """
    构建限流计数器的Redis键

    Args:
        user_id: 用户ID
        role_id: 角色ID

    Returns:
        Redis键字符串，用于存储每分钟请求计数
    """
    return build_namespaced_key("ratelimit", user_id, role_id, "minute")


def memory_summary_key(user_id: str, role_id: str, session_id: str | None = None) -> str:
    """
    构建记忆摘要的Redis键

    Args:
        user_id: 用户ID
        role_id: 角色ID
        session_id: 会话ID，可选

    Returns:
        Redis键字符串，用于存储长期记忆摘要
    """
    return build_namespaced_key("memory", user_id, role_id, _session_suffix("summary", session_id))


def ingest_status_key(user_id: str, role_id: str, doc_id: str) -> str:
    """
    构建知识库摄取状态的Redis键

    Args:
        user_id: 用户ID
        role_id: 角色ID
        doc_id: 文档ID

    Returns:
        Redis键字符串，用于存储文档摄取进度和状态
    """
    return build_namespaced_key("ingest", user_id, role_id, doc_id)


def ingest_lock_key(user_id: str, role_id: str) -> str:
    """
    构建知识库摄取锁的Redis键

    Args:
        user_id: 用户ID
        role_id: 角色ID

    Returns:
        Redis键字符串，用于实现分布式锁，防止并发摄取
    """
    return build_namespaced_key("lock", user_id, role_id, "ingest")


def query_cache_key(user_id: str, role_id: str, query: str, session_id: str | None = None) -> str:
    """
    构建查询缓存的Redis键

    Args:
        user_id: 用户ID
        role_id: 角色ID
        query: 查询文本
        session_id: 会话ID，可选

    Returns:
        Redis键字符串，使用SHA256哈希查询文本作为后缀

    Note:
        使用哈希确保键长度可控且唯一
        支持按会话隔离缓存
    """
    '''
    query.encode("utf-8")	把字符串转成字节数组  
    sha256(...)	计算 SHA-256 哈希（返回哈希对象） 256 位（bit）
    hexdigest()	把哈希转成十六进制字符串
    '''
    digest = sha256(query.encode("utf-8")).hexdigest()  # 把长问题转成固定长度哈希
    suffix = digest if not session_id else f"session:{session_id}:{digest}"
    return build_namespaced_key("cache", user_id, role_id, suffix)


def query_cache_pattern(user_id: str, role_id: str, session_id: str | None = None) -> str:
    """
    构建查询缓存的匹配模式

    Args:
        user_id: 用户ID
        role_id: 角色ID
        session_id: 会话ID，可选

    Returns:
        Redis键模式字符串，用于批量删除匹配的缓存

    Note:
        使用通配符*匹配多个键
        支持按会话清除缓存
    """
    if not session_id:
        return (
            f"{_normalize_key_segment('cache')}:"
            f"{_normalize_key_segment(user_id)}:"
            f"{_normalize_key_segment(role_id)}:*"
        )
    return (
        f"{_normalize_key_segment('cache')}:"
        f"{_normalize_key_segment(user_id)}:"
        f"{_normalize_key_segment(role_id)}:"
        f"{_normalize_key_segment(f'session:{session_id}:')}*"
    )


async def init_redis() -> None:
    """
    初始化Redis客户端连接

    Note:
        使用全局单例模式，确保只有一个Redis客户端实例
        自动测试连接是否成功
    """
    global _redis_client

    if _redis_client is not None:
        return

    settings = get_settings()
    _redis_client = Redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=settings.redis_max_connections,
    )
    await _redis_client.ping()

    logger.info(
        "redis_initialized",
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
    )


async def close_redis() -> None:
    """
    关闭Redis客户端连接

    Note:
        优雅关闭，释放所有连接
        重置全局客户端实例
    """
    global _redis_client

    if _redis_client is not None:
        await _redis_client.aclose()
        logger.info("redis_closed")

    _redis_client = None


def get_redis() -> Redis:
    """
    获取Redis客户端实例

    Returns:
        Redis客户端实例

    Raises:
        RuntimeError: 如果客户端未初始化
    """
    if _redis_client is None:
        raise RuntimeError("Redis client is not initialized. Call init_redis() first.")
    return _redis_client


async def get_redis_client() -> AsyncGenerator[Redis, None]:
    """
    异步生成器，用于依赖注入

    Yields:
        Redis客户端实例

    Note:
        主要用于FastAPI的Depends依赖注入
        确保在请求上下文中正确获取客户端
    """
    yield get_redis()
