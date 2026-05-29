# 用于 get_mysql_session 函数的返回类型注解，表示这是一个异步生成器。
from collections.abc import AsyncGenerator
'''负责 MySQL 数据库连接池的创建、管理和会话提供。'''

'''
text	                执行原始 SQL 语句（如 SELECT 1 测试连接）
AsyncEngine	            异步数据库引擎类型（连接池底层）
AsyncSession	        异步数据库会话类型（用于执行 SQL）
async_sessionmaker	    会话工厂类，用于创建 AsyncSession 实例
create_async_engine	    创建异步引擎的函数
DeclarativeBase	ORM     模型基类，所有数据库模型继承它
'''
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

'''
get_settings	获取应用配置（数据库连接 URL、连接池大小等）
get_logger	获取日志记录器
'''
from app.core.config import get_settings
from app.core.logging import get_logger
# 获取当前模块的日志记录器。
logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

# 数据库引擎（连接池），单例，初始为 None
_engine: AsyncEngine | None = None
# 会话工厂，单例，初始为 None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_mysql() -> None:
    global _engine, _session_factory

    if _engine is not None and _session_factory is not None:
        return

    settings = get_settings()
    _engine = create_async_engine(
        settings.mysql_async_dsn,
        pool_pre_ping=settings.mysql_pool_pre_ping,
        pool_size=settings.mysql_pool_size,
        max_overflow=settings.mysql_max_overflow,
        pool_recycle=settings.mysql_pool_recycle,
        future=True,
        echo=False,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    async with _engine.begin() as connection:
        await connection.execute(text("SELECT 1"))

    logger.info(
        "mysql_initialized",
        host=settings.mysql_host,
        port=settings.mysql_port,
        database=settings.mysql_database,
    )


async def close_mysql() -> None:
    global _engine, _session_factory

    if _engine is not None:
        # 关闭连接池，释放所有连接
        await _engine.dispose()
        logger.info("mysql_closed")

    _engine = None
    _session_factory = None


def get_mysql_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("MySQL engine is not initialized. Call init_mysql() first.")
    return _engine


def get_mysql_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("MySQL session factory is not initialized. Call init_mysql() first.")
    return _session_factory


async def get_mysql_session() -> AsyncGenerator[AsyncSession, None]:
    session_factory = get_mysql_session_factory()
    async with session_factory() as session:
        yield session
'''
工厂模式 = 先把"造连接的机器"建好（启动时），每次请求时用机器造一个连接，用完自动回收(退出with时)。
避免重复造机器，避免忘记回收连接。
'''