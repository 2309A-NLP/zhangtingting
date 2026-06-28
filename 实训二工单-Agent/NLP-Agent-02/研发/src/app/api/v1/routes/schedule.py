from datetime import date, time
from typing import Annotated, Literal
# FastAPI 路由文件，专门处理 日程管理（Schedule）相关的 API 请求。
from fastapi import APIRouter, Depends, Query, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
'''
导入	                        作用
date, time	               日期和时间类型（用于查询参数）
Literal	                   类型提示：限制只能取某些特定值
APIRouter	               FastAPI 路由器
Depends	                   依赖注入
Query	                   查询参数装饰器
status	                   HTTP 状态码常量
RequestValidationError	   FastAPI 的请求验证异常
ValidationError	           Pydantic 的验证异常
'''
from app.core.enums import ReminderStatus, ScheduleStatus
from app.schemas.common import ApiResponse
from app.schemas.schedule import (
    ReminderAlertLogList,
    ReminderAlertLogQuery,
    ReminderLogList,
    ReminderLogQuery,
    ReminderReliabilitySummary,
    ScheduleCreate,
    ScheduleList,
    ScheduleQuery,
    ScheduleRead,
    ScheduleSummary,
    ScheduleUpdate,
)
from app.services.schedule_service import ScheduleService, get_schedule_service

router = APIRouter()
'''
schedule_router.py
├── 导入依赖
├── 创建路由器 (APIRouter)
├── 辅助函数: get_schedule_query() (构建查询参数对象)
└── 10 个 API 端点
    ├── POST   /                      → 创建日程
    ├── GET    /                      → 列出日程（分页 + 过滤）
    ├── GET    /summary               → 日程统计摘要
    ├── GET    /reminder-logs         → 提醒日志列表
    ├── GET    /reminder-alerts       → 提醒告警日志列表
    ├── GET    /reminder-reliability  → 提醒可靠性统计
    ├── GET    /{schedule_id}         → 获取单个日程
    ├── PUT    /{schedule_id}         → 更新日程
    └── DELETE /{schedule_id}         → 删除日程
'''

async def get_schedule_query(
    query_date: Annotated[date | None, Query(alias="date")] = None,
    start_date: Annotated[date | None, Query()] = None,
    end_date: Annotated[date | None, Query()] = None,
    schedule_status: Annotated[ScheduleStatus | None, Query(alias="status")] = None,
    statuses: Annotated[list[ScheduleStatus] | None, Query()] = None,
    schedule_time_start: Annotated[time | None, Query(alias="schedule_time_start")] = None,
    schedule_time_end: Annotated[time | None, Query(alias="schedule_time_end")] = None,
    keyword: Annotated[str | None, Query(min_length=1, max_length=255)] = None,
    sort_by: Annotated[
        Literal["id", "created_at", "updated_at", "schedule_date", "schedule_time", "next_trigger_at"],
        Query(),
    ] = "created_at",
    sort_order: Annotated[Literal["asc", "desc"], Query()] = "desc",
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ScheduleQuery:
    try:
        return ScheduleQuery(
            date=query_date,
            start_date=start_date,
            end_date=end_date,
            status=schedule_status,
            statuses=statuses,
            schedule_time_start=schedule_time_start,
            schedule_time_end=schedule_time_end,
            keyword=keyword,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        )
    except ValidationError as exc:
        # 如果 Pydantic 验证失败，转成 FastAPI 的验证异常
        # 这样 FastAPI 会自动返回 422 错误和详细说明
        # 让这类参数校验错误既返回正确的 HTTP 语义，又走统一的错误响应格式
        # “异常归类”和“统一出口”
        raise RequestValidationError(exc.errors()) from exc
'''
参数	                       类型	                          说明
query_date	           date | None	                   按日期查询（URL 参数名 date）
start_date	           date | None	                   开始日期
end_date	           date | None	                   结束日期
schedule_status	       ScheduleStatus | None	       日程状态（URL 参数名 status）
statuses	           list[ScheduleStatus] | None	   多个状态（可传多次）
schedule_time_start	   time | None	                   开始时间
schedule_time_end	   time | None	                   结束时间
keyword	               str | None	                   关键词搜索（1-255 字符）
sort_by	               Literal[...]	                   排序字段（只能从列表里选）
sort_order	           Literal["asc", "desc"]	       排序方向
limit	               int	                           每页数量（1-100）
offset	               int	                           偏移量（>=0）
'''

@router.post(
    "",  # 创建日程（POST）
    response_model=ApiResponse[ScheduleRead],
    status_code=status.HTTP_201_CREATED)
async def create_schedule(
    payload: ScheduleCreate,
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
) -> ApiResponse[ScheduleRead]:
    result = await service.create(payload)
    return ApiResponse(code=201, message="created", data=result)
'''
路径：POST /（相对于前缀）
状态码：201 Created
请求体：ScheduleCreate（包含标题、日期、时间等）
依赖注入：ScheduleService 处理业务逻辑
返回：创建后的日程详情
'''

@router.get(
    "",  # 列出日程（GET）
    response_model=ApiResponse[ScheduleList],
    status_code=status.HTTP_200_OK)
async def list_schedules(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    query: Annotated[ScheduleQuery, Depends(get_schedule_query)],
) -> ApiResponse[ScheduleList]:
    result = await service.list_all(query)
    return ApiResponse(code=200, message="success", data=result)
'''
路径：GET /
查询参数：由 get_schedule_query 自动解析
分页：通过 limit 和 offset 控制
'''

@router.get(
    "/summary",  # 日程统计摘要（GET）
    response_model=ApiResponse[ScheduleSummary],
    status_code=status.HTTP_200_OK)
async def get_schedule_summary(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    query: Annotated[ScheduleQuery, Depends(get_schedule_query)],
) -> ApiResponse[ScheduleSummary]:
    result = await service.summarize(query)
    return ApiResponse(code=200, message="success", data=result)
'''
路径：GET /summary
作用：统计满足条件的日程数量（总数、待执行、已完成、已取消等）
参数：复用 ScheduleQuery，可以按日期/状态筛选
'''

@router.get(
    "/reminder-logs",   # 提醒日志列表
    response_model=ApiResponse[ReminderLogList],
    status_code=status.HTTP_200_OK)
async def list_reminder_logs(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    schedule_id: Annotated[int | None, Query()] = None,
    reminder_status: Annotated[ReminderStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[ReminderLogList]:
    query = ReminderLogQuery(
        schedule_id=schedule_id,
        status=reminder_status,
        limit=limit,
        offset=offset,
    )
    result = await service.list_reminder_logs(query)
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/reminder-alerts",  # 提醒告警日志列表
    response_model=ApiResponse[ReminderAlertLogList],
    status_code=status.HTTP_200_OK)
async def list_reminder_alert_logs(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    schedule_id: Annotated[int | None, Query()] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    alert_type: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[ReminderAlertLogList]:
    result = await service.list_alert_logs(
        ReminderAlertLogQuery(
            schedule_id=schedule_id,
            status=status_filter,
            alert_type=alert_type,
            limit=limit,
            offset=offset,
        )
    )
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/reminder-reliability",   # 提醒可靠性统计
    response_model=ApiResponse[ReminderReliabilitySummary],
    status_code=status.HTTP_200_OK)
async def get_reminder_reliability_summary(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
) -> ApiResponse[ReminderReliabilitySummary]:
    result = await service.summarize_reliability()
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/{schedule_id}",  # 获取单个日程详情
    response_model=ApiResponse[ScheduleRead],
    status_code=status.HTTP_200_OK)
async def get_schedule(
    schedule_id: int,
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
) -> ApiResponse[ScheduleRead]:
    result = await service.get_by_id(schedule_id)
    return ApiResponse(code=200, message="success", data=result)


@router.put(
    "/{schedule_id}",  # 更新日程（部分更新，ScheduleUpdate 的字段都是可选的）
    response_model=ApiResponse[ScheduleRead],
    status_code=status.HTTP_200_OK)
async def update_schedule(
    schedule_id: int,
    payload: ScheduleUpdate,
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
) -> ApiResponse[ScheduleRead]:
    result = await service.update(schedule_id, payload)
    return ApiResponse(code=200, message="updated", data=result)


@router.delete(
    "/{schedule_id}",  # 删除日程（返回被删除的 ID）
    response_model=ApiResponse[dict[str, int]],
    status_code=status.HTTP_200_OK)
async def delete_schedule(
    schedule_id: int,
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
) -> ApiResponse[dict[str, int]]:
    await service.delete(schedule_id)
    return ApiResponse(code=200, message="deleted", data={"id": schedule_id})
