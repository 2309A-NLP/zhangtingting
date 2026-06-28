from datetime import datetime
from typing import Annotated
# 调度器审计日志服务，负责记录和管理定时任务（Scheduler Job）的每次运行记录。
# 它追踪所有后台调度任务的执行情况，包括开始时间、结束时间、状态、处理数量等。
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.repositories.scheduler_audit_repository import SchedulerJobRunLogRepository
from app.schemas.scheduler_audit import (
    SchedulerJobRunLogCreate,
    SchedulerJobRunLogList,
    SchedulerJobRunLogQuery,
    SchedulerJobRunLogRead,
    SchedulerJobRuntimeSummary,
    SchedulerJobRuntimeSummaryItem,
    SchedulerJobSummary,
)

'''
SchedulerJobRunLogService
├── 初始化（依赖注入）
│   └── SchedulerJobRunLogRepository（数据访问层）
├── 工厂方法
│   └── from_session() → 从数据库会话创建 Service
├── 运行生命周期管理
│   ├── start_run() → 记录任务开始运行
│   └── finish_run() → 记录任务完成/失败
├── 日志查询
│   ├── list_logs() → 列表（分页 + 过滤）
│   └── export_logs() → 导出全部
└── 统计分析
    ├── summarize() → 全局统计摘要
    └── summarize_runtime() → 按任务分组统计
'''

class SchedulerJobRunLogService:
    def __init__(self, repository: SchedulerJobRunLogRepository) -> None:
        self._repository = repository

    @classmethod
    def from_session(cls, session: AsyncSession) -> "SchedulerJobRunLogService":
        return cls(SchedulerJobRunLogRepository(session))

    # 记录任务开始
    async def start_run(
        self,
        *,
        job_id: str,
        job_name: str,
        trigger_name: str | None,
        started_at: datetime,
    ) -> SchedulerJobRunLogRead:
        record = await self._repository.create(
            SchedulerJobRunLogCreate(
                job_id=job_id,
                job_name=job_name,
                trigger_name=trigger_name,
                started_at=started_at,
                finished_at=None,
                status="running",
                processed_count=0,
                error_message=None,
            )
        )
        return SchedulerJobRunLogRead.model_validate(record)

    # 记录任务完成
    async def finish_run(
        self,
        run_log: SchedulerJobRunLogRead,
        *,
        finished_at: datetime,
        status: str,
        processed_count: int,
        error_message: str | None,
    ) -> SchedulerJobRunLogRead:
        record = await self._repository.get_by_id(run_log.id)
        if record is None:
            raise ValueError(f"Scheduler job run log {run_log.id} not found")
        record = await self._repository.update_run(
            record,
            finished_at=finished_at,
            status=status,
            processed_count=processed_count,
            error_message=error_message,
        )
        return SchedulerJobRunLogRead.model_validate(record)

    # 查询日志列表
    async def list_logs(
        self,
        *,
        job_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> SchedulerJobRunLogList:
        query = SchedulerJobRunLogQuery(job_id=job_id, status=status, limit=limit, offset=offset)
        records, total = await self._repository.list_logs(query)
        return SchedulerJobRunLogList(
            total=total,
            limit=limit,
            offset=offset,
            items=[SchedulerJobRunLogRead.model_validate(item) for item in records],
        )

    # 导出日志
    async def export_logs(
        self,
        *,
        job_id: str | None = None,
        status: str | None = None,
    ) -> list[SchedulerJobRunLogRead]:
        query = SchedulerJobRunLogQuery(job_id=job_id, status=status, limit=20, offset=0)
        records = await self._repository.export_logs(query)
        return [SchedulerJobRunLogRead.model_validate(item) for item in records]

    # 全局统计
    async def summarize(self) -> SchedulerJobSummary:
        summary = await self._repository.summarize()
        return SchedulerJobSummary(**summary)

    # 按任务分组统计
    async def summarize_runtime(self) -> SchedulerJobRuntimeSummary:
        items_raw = await self._repository.summarize_by_job()
        items = [SchedulerJobRuntimeSummaryItem(**item) for item in items_raw]
        return SchedulerJobRuntimeSummary(total_jobs=len(items), items=items)


async def get_scheduler_audit_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> SchedulerJobRunLogService:
    return SchedulerJobRunLogService.from_session(session)
