from sqlalchemy import Select, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schedule import LLMAuditLog
from app.schemas.llm_audit import LLMAuditLogCreate, LLMAuditLogQuery


class LLMAuditLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, payload: LLMAuditLogCreate) -> LLMAuditLog:
        record = LLMAuditLog(**payload.model_dump())
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def list_by_session_id(self, session_id: str) -> list[LLMAuditLog]:
        statement: Select[tuple[LLMAuditLog]] = (
            select(LLMAuditLog)
            .where(LLMAuditLog.session_id == session_id)
            .order_by(LLMAuditLog.id.asc())
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def list_logs(self, query: LLMAuditLogQuery) -> tuple[list[LLMAuditLog], int]:
        statement: Select[tuple[LLMAuditLog]] = select(LLMAuditLog)
        count_statement = select(func.count()).select_from(LLMAuditLog)

        if query.session_id is not None:
            statement = statement.where(LLMAuditLog.session_id == query.session_id)
            count_statement = count_statement.where(LLMAuditLog.session_id == query.session_id)
        if query.parser_stage is not None:
            statement = statement.where(LLMAuditLog.parser_stage == query.parser_stage)
            count_statement = count_statement.where(LLMAuditLog.parser_stage == query.parser_stage)
        if query.success is not None:
            statement = statement.where(LLMAuditLog.success == query.success)
            count_statement = count_statement.where(LLMAuditLog.success == query.success)

        statement = statement.order_by(LLMAuditLog.id.desc()).offset(query.offset).limit(query.limit)
        result = await self._session.execute(statement)
        count_result = await self._session.execute(count_statement)
        return list(result.scalars().all()), int(count_result.scalar_one())

    async def export_logs(self, query: LLMAuditLogQuery) -> list[LLMAuditLog]:
        statement: Select[tuple[LLMAuditLog]] = select(LLMAuditLog)

        if query.session_id is not None:
            statement = statement.where(LLMAuditLog.session_id == query.session_id)
        if query.parser_stage is not None:
            statement = statement.where(LLMAuditLog.parser_stage == query.parser_stage)
        if query.success is not None:
            statement = statement.where(LLMAuditLog.success == query.success)

        statement = statement.order_by(LLMAuditLog.id.desc())
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def summarize(self) -> dict[str, int]:
        statement = select(
            func.count().label("total"),
            func.coalesce(func.sum(case((LLMAuditLog.success.is_(True), 1), else_=0)), 0).label("success_count"),
            func.coalesce(func.sum(case((LLMAuditLog.success.is_(False), 1), else_=0)), 0).label("failed_count"),
            func.coalesce(func.sum(case((LLMAuditLog.parser_stage == "repair", 1), else_=0)), 0).label("repair_count"),
        ).select_from(LLMAuditLog)

        result = await self._session.execute(statement)
        row = result.one()
        return {
            "total": int(row.total),
            "success_count": int(row.success_count),
            "failed_count": int(row.failed_count),
            "repair_count": int(row.repair_count),
        }
