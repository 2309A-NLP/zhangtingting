from typing import Annotated
# 调度器审计（Scheduler Audit）路由文件
# 这是一个调度器审计路由文件，提供任务执行日志、统计摘要、性能分析和分布式锁状态监控功能。它让管理员能随时查看定时任务跑得怎么样、有没有报错、有没有卡住，是运维调度器的核心监控工具。
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import Response

from app.core.auth import require_admin_access
from app.schemas.common import ApiResponse
from app.schemas.scheduler_audit import (
    SchedulerJobRunLogList,
    SchedulerJobRuntimeSummary,
    SchedulerJobSummary,
)
from app.schemas.scheduler_runtime import SchedulerLeaseRead
from app.services.scheduler_audit_service import (
    SchedulerJobRunLogService,
    get_scheduler_audit_service,
)
from app.services.scheduler_lock_service import SchedulerLockService, get_scheduler_lock_service
from app.utils.csv_export import build_csv_bytes

'''
prefix="/admin"：所有路由加 /admin 前缀
dependencies=[Depends(require_admin_access)]：所有端点都需要管理员认证
tags=["scheduler-audit"]：Swagger 文档中分组为 "scheduler-audit"
'''
router = APIRouter(prefix="/admin", tags=["scheduler-audit"], dependencies=[Depends(require_admin_access)])
'''
scheduler_audit_router.py
├── 导入依赖
├── 创建路由器 (prefix="/admin", dependencies=[Depends(require_admin_access)])
├── 辅助函数: _csv_response() (CSV 导出)
└── 5 个 API 端点
    ├── GET /scheduler/jobs           → 任务执行日志列表
    ├── GET /scheduler/jobs/export    → 导出任务日志 CSV
    ├── GET /scheduler/summary        → 任务执行统计摘要
    ├── GET /scheduler/runtime-summary → 任务运行时统计
    └── GET /scheduler/leases/{job_id} → 获取任务分布式锁状态
'''

def _csv_response(*, filename_prefix: str, rows: list[dict[str, object]], fieldnames: list[str]) -> Response:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_bytes = build_csv_bytes(rows, fieldnames)
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename_prefix}_{timestamp}.csv"'},
    )


@router.get(
    "/scheduler/jobs",  # 任务执行日志列表（GET）
    response_model=ApiResponse[SchedulerJobRunLogList],
    status_code=status.HTTP_200_OK)
async def list_scheduler_job_runs(
    service: Annotated[SchedulerJobRunLogService, Depends(get_scheduler_audit_service)],
    job_id: Annotated[str | None, Query()] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[SchedulerJobRunLogList]:
    result = await service.list_logs(job_id=job_id, status=status_filter, limit=limit, offset=offset)
    return ApiResponse(code=200, message="success", data=result)
'''
作用： 查看调度器任务的所有执行记录
参数说明：
参数	                  说明
job_id	           按任务 ID 过滤
status_filter	   按状态过滤（success / failure / running）
limit	           每页数量（1-200）
offset	           分页偏移量
'''

@router.get(
    "/scheduler/jobs/export",  # 导出任务日志 CSV（GET）
    response_class=Response,
    status_code=status.HTTP_200_OK)
async def export_scheduler_job_runs(
    service: Annotated[SchedulerJobRunLogService, Depends(get_scheduler_audit_service)],
    job_id: Annotated[str | None, Query()] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
) -> Response:
    records = await service.export_logs(job_id=job_id, status=status_filter)
    rows = [item.model_dump(mode="json") for item in records]
    return _csv_response(
        filename_prefix="scheduler_job_runs",
        rows=rows,
        fieldnames=[
            "id",   # 日志记录 ID
            "job_id",   # 任务唯一标识
            "job_name",   # 任务名称
            "trigger_name",  # 触发器名称
            "started_at",   # 开始时间
            "finished_at",   # 结束时间
            "status",        # 执行状态
            "processed_count",  # 处理数量
            "error_message",    # 错误信息
            "created_at",         # 创建时间
            "updated_at",      # 更新时间
        ],
    )


@router.get(
    "/scheduler/summary",  # 任务执行统计摘要（GET）
    response_model=ApiResponse[SchedulerJobSummary],
    status_code=status.HTTP_200_OK)
async def get_scheduler_job_summary(
    service: Annotated[SchedulerJobRunLogService, Depends(get_scheduler_audit_service)],
) -> ApiResponse[SchedulerJobSummary]:
    result = await service.summarize()
    return ApiResponse(code=200, message="success", data=result)
# 获取所有任务的执行统计摘要

@router.get(
    "/scheduler/runtime-summary",  # 任务运行时统计（GET）
    response_model=ApiResponse[SchedulerJobRuntimeSummary],
    status_code=status.HTTP_200_OK,
)
async def get_scheduler_runtime_summary(
    service: Annotated[SchedulerJobRunLogService, Depends(get_scheduler_audit_service)],
) -> ApiResponse[SchedulerJobRuntimeSummary]:
    result = await service.summarize_runtime()
    return ApiResponse(code=200, message="success", data=result)
# 作用： 获取任务的运行时统计（执行时间分布、最慢任务等）

'''
端点	                        关注点	数据
/scheduler/summary	        执行结果	成功/失败数量、成功率
/scheduler/runtime-summary	执行性能	执行时间、最慢任务、平均耗时
'''

@router.get(
    "/scheduler/leases/{job_id}",  # 获取任务分布式锁状态（GET）
    response_model=ApiResponse[SchedulerLeaseRead | None],
    status_code=status.HTTP_200_OK)
async def get_scheduler_job_lease(
    job_id: str,
    service: Annotated[SchedulerLockService, Depends(get_scheduler_lock_service)],
) -> ApiResponse[SchedulerLeaseRead | None]:
    result = await service.get_lease(job_id)
    return ApiResponse(code=200, message="success", data=result)
# 作用： 检查某个任务是否被"锁定"（正在执行中）