from __future__ import annotations
'''审计日志服务，用于记录系统中所有重要操作的踪迹，方便追溯和排查问题。
审计日志 = 谁 + 什么时候 + 做了什么 + 结果如何'''
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger

logger = get_logger(__name__)


class AuditService:
    async def write_log(
        self,
        db_session: AsyncSession,
        *,
        action: str,  # 操作类型
        request_id: str | None = None,
        user_id: str | None = None,
        role_id: str | None = None,
        resource_type: str | None = None,  # 资源类型
        resource_id: str | None = None,
        status: str = "success",
        message: str | None = None,
    ) -> None:
        stmt = text(
            """
            INSERT INTO audit_logs (request_id, user_id, role_id, action, resource_type, resource_id, status, message, created_at)
            VALUES (:request_id, :user_id, :role_id, :action, :resource_type, :resource_id, :status, :message, NOW())
            """
        )
        await db_session.execute(
            stmt,
            {
                "request_id": request_id,
                "user_id": user_id,
                "role_id": role_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "status": status,
                "message": message,
            },
        )
        await db_session.commit()
        logger.info("audit_log_written", action=action, resource_id=resource_id, status=status)
