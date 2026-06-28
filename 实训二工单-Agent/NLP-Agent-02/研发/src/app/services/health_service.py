from typing import Annotated
from urllib.parse import urlsplit
# urlsplit：解析数据库 URL，提取 scheme（协议类型）
# text()：SQLAlchemy 原生 SQL 执行器
from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db_session
from app.queue import ReminderQueueBackend, get_reminder_queue_backend
from app.schemas.health import HealthDatabaseStatus, HealthReadinessStatus, HealthRedisStatus


class HealthService:
    def __init__(
        self,
        session: AsyncSession,
        queue_backend: ReminderQueueBackend | None = None,
    ) -> None:
        self._session = session
        self._queue_backend = queue_backend or get_reminder_queue_backend()

    async def get_readiness(self) -> HealthReadinessStatus:
        # 执行一个最简单的 SQL 查询，验证数据库连接是否正常
        await self._session.execute(text("SELECT 1"))
        # 解析数据库 URL：
        # settings.database_url 示例：postgresql://user:pass@localhost:5432/db
        # urlsplit() 解析后提取 scheme → "postgresql"
        # 如果没有 scheme，默认 "unknown"
        # 用途：返回给前端，展示当前使用的数据库类型（postgresql/mysql/sqlite 等）
        scheme = urlsplit(settings.database_url).scheme or "unknown"
        redis_connected = False
        redis_backlog = 0
        if self._queue_backend.enabled:
            try:
                redis_backlog = await self._queue_backend.get_queue_length()
                redis_connected = True
            except Exception:  # pragma: no cover
                redis_connected = False
        return HealthReadinessStatus(
            status="ok",
            role=settings.app_role,
            database=HealthDatabaseStatus(
                connected=True,
                database_url_scheme=scheme,
            ),
            redis=HealthRedisStatus(
                enabled=self._queue_backend.enabled,
                connected=redis_connected,
                queue_backlog=redis_backlog,
            ),
        )


async def get_health_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> HealthService:
    return HealthService(session, queue_backend=get_reminder_queue_backend())
