from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schedule import AdminAccessAuditLog
from app.schemas.admin_audit import AdminAccessAuditLogCreate


class AdminAccessAuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, payload: AdminAccessAuditLogCreate) -> AdminAccessAuditLog:
        record = AdminAccessAuditLog(**payload.model_dump())
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def list_logs(self, *, limit: int, offset: int) -> tuple[list[AdminAccessAuditLog], int]:
        statement: Select[tuple[AdminAccessAuditLog]] = (
            select(AdminAccessAuditLog)
            .order_by(AdminAccessAuditLog.id.desc())
            .offset(offset)
            .limit(limit)
        )
        count_statement = select(func.count()).select_from(AdminAccessAuditLog)
        result = await self._session.execute(statement)
        count_result = await self._session.execute(count_statement)
        return list(result.scalars().all()), int(count_result.scalar_one())

    async def export_logs(self) -> list[AdminAccessAuditLog]:
        statement: Select[tuple[AdminAccessAuditLog]] = select(AdminAccessAuditLog).order_by(AdminAccessAuditLog.id.desc())
        result = await self._session.execute(statement)
        return list(result.scalars().all())
