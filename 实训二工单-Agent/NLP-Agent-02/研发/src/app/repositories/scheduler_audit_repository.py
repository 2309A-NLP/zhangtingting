from datetime import datetime

from sqlalchemy import Select, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schedule import SchedulerJobRunLog
from app.schemas.scheduler_audit import SchedulerJobRunLogCreate, SchedulerJobRunLogQuery


class SchedulerJobRunLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, log_id: int) -> SchedulerJobRunLog | None:
        result = await self._session.execute(
            select(SchedulerJobRunLog).where(SchedulerJobRunLog.id == log_id)
        )
        return result.scalar_one_or_none()

    async def create(self, payload: SchedulerJobRunLogCreate) -> SchedulerJobRunLog:
        record = SchedulerJobRunLog(**payload.model_dump())
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def update_run(
        self,
        record: SchedulerJobRunLog,
        *,
        finished_at: datetime,
        status: str,
        processed_count: int,
        error_message: str | None,
    ) -> SchedulerJobRunLog:
        record.finished_at = finished_at
        record.status = status
        record.processed_count = processed_count
        record.error_message = error_message
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def list_logs(self, query: SchedulerJobRunLogQuery) -> tuple[list[SchedulerJobRunLog], int]:
        statement: Select[tuple[SchedulerJobRunLog]] = select(SchedulerJobRunLog)
        count_statement = select(func.count()).select_from(SchedulerJobRunLog)

        if query.job_id is not None:
            statement = statement.where(SchedulerJobRunLog.job_id == query.job_id)
            count_statement = count_statement.where(SchedulerJobRunLog.job_id == query.job_id)
        if query.status is not None:
            statement = statement.where(SchedulerJobRunLog.status == query.status)
            count_statement = count_statement.where(SchedulerJobRunLog.status == query.status)

        statement = statement.order_by(SchedulerJobRunLog.id.desc()).offset(query.offset).limit(query.limit)
        result = await self._session.execute(statement)
        count_result = await self._session.execute(count_statement)
        return list(result.scalars().all()), int(count_result.scalar_one())

    async def export_logs(self, query: SchedulerJobRunLogQuery) -> list[SchedulerJobRunLog]:
        statement: Select[tuple[SchedulerJobRunLog]] = select(SchedulerJobRunLog)

        if query.job_id is not None:
            statement = statement.where(SchedulerJobRunLog.job_id == query.job_id)
        if query.status is not None:
            statement = statement.where(SchedulerJobRunLog.status == query.status)

        statement = statement.order_by(SchedulerJobRunLog.id.desc())
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def summarize(self) -> dict[str, int]:
        statement = select(
            func.count().label("total"),
            func.coalesce(
                func.sum(case((SchedulerJobRunLog.status == "running", 1), else_=0)),
                0,
            ).label("running_count"),
            func.coalesce(
                func.sum(case((SchedulerJobRunLog.status == "success", 1), else_=0)),
                0,
            ).label("success_count"),
            func.coalesce(
                func.sum(case((SchedulerJobRunLog.status == "failed", 1), else_=0)),
                0,
            ).label("failed_count"),
        ).select_from(SchedulerJobRunLog)

        result = await self._session.execute(statement)
        row = result.one()
        return {
            "total": int(row.total),
            "running_count": int(row.running_count),
            "success_count": int(row.success_count),
            "failed_count": int(row.failed_count),
        }

    async def summarize_by_job(self) -> list[dict[str, object]]:
        jobs_result = await self._session.execute(
            select(SchedulerJobRunLog.job_id).distinct().order_by(SchedulerJobRunLog.job_id.asc())
        )
        job_ids = [str(item[0]) for item in jobs_result.all()]
        summaries: list[dict[str, object]] = []

        for job_id in job_ids:
            aggregate_result = await self._session.execute(
                select(
                    func.count().label("total_runs"),
                    func.coalesce(
                        func.sum(case((SchedulerJobRunLog.status == "running", 1), else_=0)),
                        0,
                    ).label("running_count"),
                    func.coalesce(
                        func.sum(case((SchedulerJobRunLog.status == "success", 1), else_=0)),
                        0,
                    ).label("success_count"),
                    func.coalesce(
                        func.sum(case((SchedulerJobRunLog.status == "failed", 1), else_=0)),
                        0,
                    ).label("failed_count"),
                    func.coalesce(func.sum(SchedulerJobRunLog.processed_count), 0).label("total_processed_count"),
                    func.coalesce(func.avg(SchedulerJobRunLog.processed_count), 0).label("avg_processed_count"),
                ).where(SchedulerJobRunLog.job_id == job_id)
            )
            aggregate_row = aggregate_result.one()

            latest_result = await self._session.execute(
                select(SchedulerJobRunLog)
                .where(SchedulerJobRunLog.job_id == job_id)
                .order_by(SchedulerJobRunLog.id.desc())
                .limit(1)
            )
            latest = latest_result.scalar_one()
            summaries.append(
                {
                    "job_id": job_id,
                    "total_runs": int(aggregate_row.total_runs),
                    "running_count": int(aggregate_row.running_count),
                    "success_count": int(aggregate_row.success_count),
                    "failed_count": int(aggregate_row.failed_count),
                    "total_processed_count": int(aggregate_row.total_processed_count),
                    "avg_processed_count": round(float(aggregate_row.avg_processed_count), 2),
                    "last_status": latest.status,
                    "last_started_at": latest.started_at,
                    "last_finished_at": latest.finished_at,
                    "last_error_message": latest.error_message,
                }
            )

        return summaries
