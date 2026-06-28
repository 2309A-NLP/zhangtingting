from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.repositories.admin_audit_repository import AdminAccessAuditRepository
from app.schemas.admin_audit import (
    AdminAccessAuditLogCreate,
    AdminAccessAuditLogList,
    AdminAccessAuditLogRead,
)


class AdminAccessAuditService:
    def __init__(self, repository: AdminAccessAuditRepository) -> None:
        self._repository = repository

    @classmethod
    def from_session(cls, session: AsyncSession) -> "AdminAccessAuditService":
        return cls(AdminAccessAuditRepository(session))

    async def record(self, payload: AdminAccessAuditLogCreate) -> AdminAccessAuditLogRead:
        record = await self._repository.create(payload)
        return AdminAccessAuditLogRead.model_validate(record)

    async def list_logs(self, *, limit: int = 20, offset: int = 0) -> AdminAccessAuditLogList:
        records, total = await self._repository.list_logs(limit=limit, offset=offset)
        return AdminAccessAuditLogList(
            total=total,
            limit=limit,
            offset=offset,
            items=[AdminAccessAuditLogRead.model_validate(item) for item in records],
        )

    async def export_logs(self) -> list[AdminAccessAuditLogRead]:
        records = await self._repository.export_logs()
        return [AdminAccessAuditLogRead.model_validate(item) for item in records]


async def get_admin_audit_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> AdminAccessAuditService:
    return AdminAccessAuditService.from_session(session)
