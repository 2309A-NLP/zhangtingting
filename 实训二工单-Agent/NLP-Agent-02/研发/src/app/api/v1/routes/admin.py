from datetime import datetime
from typing import Annotated
# FastAPI 管理员路由文件，专门为管理员（Admin）提供后台管理功能。
# 它比普通用户路由有更高的权限，并且提供了数据导出（CSV）功能。
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import Response

from app.api.v1.routes.schedule import get_schedule_query
from app.core.auth import require_admin_access
from app.core.enums import ReminderStatus
from app.schemas.admin_audit import AdminAccessAuditLogList
from app.schemas.common import ApiResponse
from app.schemas.conversation import ConversationSessionList, ConversationSessionView
from app.schemas.conversation_history import ConversationHistoryList, ConversationHistoryRead
from app.schemas.dashboard import DashboardOverview
from app.schemas.llm_audit import LLMAuditLogList, LLMAuditLogRead
from app.schemas.schedule import (
    ReminderAlertLogList,
    ReminderAlertLogQuery,
    ReminderDeliveryQueueSummary,
    ReminderDeliveryTaskList,
    ReminderDeliveryTaskQuery,
    ReminderLogList,
    ReminderLogQuery,
    ReminderReliabilitySummary,
    ScheduleList,
    ScheduleQuery,
    ScheduleSummary,
)
from app.services.admin_audit_service import AdminAccessAuditService, get_admin_audit_service
from app.services.conversation_history_service import (
    ConversationHistoryService,
    get_conversation_history_service,
)
from app.services.conversation_service import ConversationService, get_conversation_service
from app.services.dashboard_service import DashboardService, get_dashboard_service
from app.services.llm_audit_service import LLMAuditLogService, get_llm_audit_log_service
from app.services.schedule_service import ScheduleService, get_schedule_service
from app.utils.csv_export import build_csv_bytes

'''
参数	                                                含义
prefix="/admin"	                                所有路由都加 /admin 前缀
tags=["admin"]	                                在 Swagger 文档中分组为 "admin"
dependencies=[Depends(require_admin_access)]	所有端点都需要管理员认证！
'''
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin_access)])
'''
admin_router.py
├── 导入依赖
├── 创建路由器 (prefix="/admin", dependencies=[Depends(require_admin_access)])
├── 辅助函数: _csv_response() (生成 CSV 下载响应)
└── 23 个 API 端点
    ├── 仪表盘 (1)
    │   └── GET /dashboard/overview → 系统总览
    ├── 日程管理 (10)
    │   ├── GET    /schedule              → 列出日程
    │   ├── GET    /schedule/export       → 导出日程 CSV
    │   ├── GET    /schedule/summary      → 日程统计
    │   ├── GET    /schedule/reminder-logs → 提醒日志
    │   ├── GET    /schedule/reminder-logs/export → 导出提醒日志
    │   ├── GET    /schedule/reminder-alerts → 告警日志
    │   ├── GET    /schedule/reminder-alerts/export → 导出告警日志
    │   ├── GET    /schedule/reminder-reliability → 可靠性统计
    │   ├── GET    /schedule/delivery-tasks → 投递任务列表
    │   ├── GET    /schedule/delivery-tasks/export → 导出投递任务
    │   ├── GET    /schedule/delivery-queue → 投递队列摘要
    │   ├── POST   /schedule/delivery-tasks/{id}/retry → 重试投递
    │   └── POST   /schedule/delivery-tasks/{id}/unlock → 解锁任务
    ├── Agent 管理 (7)
    │   ├── GET /agent/sessions           → 会话列表
    │   ├── GET /agent/sessions/history   → 会话历史（带过滤）
    │   ├── GET /agent/sessions/history/export → 导出历史
    │   ├── GET /agent/sessions/{id}/history → 单个会话历史
    │   ├── GET /agent/sessions/{id}      → 单个会话详情
    │   ├── GET /agent/llm-audit          → LLM 审计日志
    │   ├── GET /agent/llm-audit/export   → 导出 LLM 审计
    │   └── GET /agent/llm-audit/{id}     → 单个会话审计
    └── 访问审计 (2)
        ├── GET /access-audit             → 访问审计日志
        └── GET /access-audit/export      → 导出访问审计
'''

# 生成 CSV 文件下载响应
def _csv_response(*, filename_prefix: str, rows: list[dict[str, object]], fieldnames: list[str]) -> Response:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    csv_bytes = build_csv_bytes(rows, fieldnames)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    '''
    Content-Disposition：HTTP 响应头，告诉浏览器如何处理返回的内容。
    attachment：表示内容应作为附件下载，而不是在浏览器中直接打开（比如预览 PDF 或图片）。
    filename="report_20260624_143025.csv"：指定下载后的默认文件名。
    效果：浏览器收到这个头后，会弹出"保存文件"对话框，默认文件名就是这里指定的值。
    '''
    '''
    content=csv_bytes：响应体内容，即 CSV 文件的字节数据。
    media_type="text/csv; charset=utf-8"：
    设置 Content-Type 响应头为 text/csv
    明确指定编码为 UTF-8，保证中文等特殊字符正确显示
    headers=headers：附加自定义头（前面设置的 Content-Disposition）
    返回 Response 对象：FastAPI 会自动将 Response 对象作为 HTTP 响应发送给客户端。
    '''
    return Response(content=csv_bytes, media_type="text/csv; charset=utf-8", headers=headers)
'''
参数	               说明
filename_prefix	  文件名前缀（如 admin_schedules）
rows	          数据行列表（每个元素是一个 dict）
fieldnames	      CSV 列头（按这个顺序输出）
'''

@router.get(
    "/dashboard/overview", # 仪表盘总览（GET）
    response_model=ApiResponse[DashboardOverview],
    status_code=status.HTTP_200_OK)
async def get_admin_dashboard_overview(
    service: Annotated[DashboardService, Depends(get_dashboard_service)],
) -> ApiResponse[DashboardOverview]:
    result = await service.get_overview()
    return ApiResponse(code=200, message="success", data=result)
'''
作用： 管理员首页的数据总览
总日程数
待执行日程数
提醒成功率
活跃会话数
LLM 调用统计
'''

@router.get(
    "/schedule",  # 列出日程
    response_model=ApiResponse[ScheduleList],
    status_code=status.HTTP_200_OK)
async def list_admin_schedules(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    query: Annotated[ScheduleQuery, Depends(get_schedule_query)],
) -> ApiResponse[ScheduleList]:
    result = await service.list_all(query)
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/schedule/export", # 导出日程 CSV
    response_class=Response,
    status_code=status.HTTP_200_OK)
async def export_admin_schedules(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    query: Annotated[ScheduleQuery, Depends(get_schedule_query)],
) -> Response:
    records = await service.export_all(query)
    # model_dump() 是 Pydantic V2 的核心方法，用于将 Pydantic 模型序列化为字典。
    # mode="json" 的作用  将复杂类型转换为 JSON 兼容的类型
    rows = [item.model_dump(mode="json") for item in records]
    return _csv_response(
        filename_prefix="admin_schedules",
        rows=rows,
        fieldnames=[
            "id",
            "content",
            "schedule_date",
            "schedule_time",
            "cycle_rule",
            "cycle_value",
            "source_text",
            "status",
            "next_trigger_at",
            "created_at",
            "updated_at",
        ],
    )


@router.get(
    "/schedule/summary", # 日程统计
    response_model=ApiResponse[ScheduleSummary],
    status_code=status.HTTP_200_OK)
async def get_admin_schedule_summary(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    query: Annotated[ScheduleQuery, Depends(get_schedule_query)],
) -> ApiResponse[ScheduleSummary]:
    result = await service.summarize(query)
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/schedule/reminder-logs",  # 提醒日志列表
    response_model=ApiResponse[ReminderLogList],
    status_code=status.HTTP_200_OK)
async def list_admin_reminder_logs(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    schedule_id: Annotated[int | None, Query()] = None,
    reminder_status: Annotated[ReminderStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[ReminderLogList]:
    result = await service.list_reminder_logs(
        ReminderLogQuery(
            schedule_id=schedule_id,
            status=reminder_status,
            limit=limit,
            offset=offset,
        )
    )
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/schedule/reminder-logs/export",  # 导出提醒日志
    response_class=Response,
    status_code=status.HTTP_200_OK)
async def export_admin_reminder_logs(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    schedule_id: Annotated[int | None, Query()] = None,
    reminder_status: Annotated[ReminderStatus | None, Query(alias="status")] = None,
) -> Response:
    records = await service.export_reminder_logs(
        ReminderLogQuery(
            schedule_id=schedule_id,
            status=reminder_status,
        )
    )
    rows = [item.model_dump(mode="json") for item in records]
    return _csv_response(
        filename_prefix="admin_reminder_logs",
        rows=rows,
        fieldnames=[
            "id",
            "schedule_id",
            "planned_trigger_at",
            "reminded_at",
            "status",
            "error_message",
            "created_at",
            "updated_at",
        ],
    )


@router.get(
    "/schedule/reminder-alerts",  # 告警日志列表
    response_model=ApiResponse[ReminderAlertLogList],
    status_code=status.HTTP_200_OK)
async def list_admin_reminder_alert_logs(
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
    "/schedule/reminder-alerts/export",  # 导出告警日志
    response_class=Response,
    status_code=status.HTTP_200_OK)
async def export_admin_reminder_alert_logs(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    schedule_id: Annotated[int | None, Query()] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    alert_type: Annotated[str | None, Query()] = None,
) -> Response:
    records = await service.export_alert_logs(
        ReminderAlertLogQuery(
            schedule_id=schedule_id,
            status=status_filter,
            alert_type=alert_type,
        )
    )
    rows = [item.model_dump(mode="json") for item in records]
    return _csv_response(
        filename_prefix="admin_reminder_alerts",
        rows=rows,
        fieldnames=[
            "id",
            "schedule_id",
            "reminder_log_id",
            "alert_type",
            "alert_channel",
            "status",
            "message",
            "sent_at",
            "error_message",
            "created_at",
            "updated_at",
        ],
    )


@router.get(
    "/schedule/reminder-reliability",  # 提醒可靠性统计
    response_model=ApiResponse[ReminderReliabilitySummary],
    status_code=status.HTTP_200_OK)
async def get_admin_reminder_reliability(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
) -> ApiResponse[ReminderReliabilitySummary]:
    result = await service.summarize_reliability()
    return ApiResponse(code=200, message="success", data=result)

'''
提醒发送
    ↓
成功 → 记录到 reminder-logs（成功状态）
    ↓
失败 → 记录到 reminder-alerts（告警）
    ↓
统计 → reminder-reliability（成功率、失败率）
'''

@router.get(
    "/schedule/delivery-tasks", # 投递任务列表
    response_model=ApiResponse[ReminderDeliveryTaskList],
    status_code=status.HTTP_200_OK)
async def list_admin_delivery_tasks(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    schedule_id: Annotated[int | None, Query()] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[ReminderDeliveryTaskList]:
    result = await service.list_delivery_tasks(
        ReminderDeliveryTaskQuery(
            schedule_id=schedule_id,
            status=status_filter,
            limit=limit,
            offset=offset,
        )
    )
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/schedule/delivery-tasks/export", # 导出投递任务
    response_class=Response,
    status_code=status.HTTP_200_OK)
async def export_admin_delivery_tasks(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
    schedule_id: Annotated[int | None, Query()] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
) -> Response:
    records = await service.export_delivery_tasks(
        ReminderDeliveryTaskQuery(
            schedule_id=schedule_id,
            status=status_filter,
        )
    )
    rows = [item.model_dump(mode="json") for item in records]
    return _csv_response(
        filename_prefix="admin_delivery_tasks",
        rows=rows,
        fieldnames=[
            "id",
            "schedule_id",
            "reminder_log_id",
            "task_type",
            "status",
            "available_at",
            "locked_by",
            "locked_at",
            "last_error_message",
            "created_at",
            "updated_at",
        ],
    )


@router.get(
    "/schedule/delivery-queue", # 投递队列摘要
    response_model=ApiResponse[ReminderDeliveryQueueSummary],
    status_code=status.HTTP_200_OK)
async def get_admin_delivery_queue_summary(
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
) -> ApiResponse[ReminderDeliveryQueueSummary]:
    result = await service.summarize_delivery_queue()
    return ApiResponse(code=200, message="success", data=result)


@router.post(
    "/schedule/delivery-tasks/{task_id}/retry",  # 重试投递任务
    response_model=ApiResponse[dict[str, object]],
    status_code=status.HTTP_200_OK,
)
async def retry_admin_delivery_task(
    task_id: int,
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
) -> ApiResponse[dict[str, object]]:
    result = await service.retry_delivery_task(task_id)
    return ApiResponse(
        code=200,
        message="retried",
        data={"task_id": task_id, "status": result.status, "available_at": result.available_at},
    )


@router.post(
    "/schedule/delivery-tasks/{task_id}/unlock",  # 解锁投递任务
    response_model=ApiResponse[dict[str, object]],
    status_code=status.HTTP_200_OK,
)
async def unlock_admin_delivery_task(
    task_id: int,
    service: Annotated[ScheduleService, Depends(get_schedule_service)],
) -> ApiResponse[dict[str, object]]:
    result = await service.unlock_delivery_task(task_id)
    return ApiResponse(
        code=200,
        message="unlocked",
        data={"task_id": task_id, "status": result.status, "available_at": result.available_at},
    )

'''
任务创建 → pending（待处理）
    ↓
调度器获取 → locked（被锁定）
    ↓
执行完成 → completed（已完成）
    ↓ 失败
error（错误）→ retry（重试）→ pending
    ↓ 锁超时
unlock（解锁）→ pending（重新进入队列）

管理员操作：
重试：手动重新执行失败的任务
解锁：如果任务被锁定但长时间未完成，可以手动解锁
'''

@router.get(
    "/agent/sessions",  # 会话列表
    response_model=ApiResponse[ConversationSessionList],
    status_code=status.HTTP_200_OK)
async def list_admin_agent_sessions(
    service: Annotated[ConversationService, Depends(get_conversation_service)],
    include_expired: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[ConversationSessionList]:
    result = await service.list_sessions(
        include_expired=include_expired,
        limit=limit,
        offset=offset,
    )
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/agent/sessions/history",  # 会话历史（带过滤）
    response_model=ApiResponse[ConversationHistoryList],
    status_code=status.HTTP_200_OK,
)
async def list_admin_agent_session_history(
    service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
    session_id: Annotated[str | None, Query()] = None,
    parser_source: Annotated[str | None, Query()] = None,
    agent_state: Annotated[str | None, Query()] = None,
    intent: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[ConversationHistoryList]:
    result = await service.list_logs(
        session_id=session_id,
        parser_source=parser_source,
        agent_state=agent_state,
        intent=intent,
        limit=limit,
        offset=offset,
    )
    return ApiResponse(code=200, message="success", data=result)
# 根据会话ID、数据来源、智能体状态、用户意图等条件，分页查询历史对话记录。

@router.get(
    "/agent/sessions/history/export",  # 导出会话历史
    response_class=Response,
    status_code=status.HTTP_200_OK)
async def export_admin_agent_session_history(
    service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
    session_id: Annotated[str | None, Query()] = None,
    parser_source: Annotated[str | None, Query()] = None,
    agent_state: Annotated[str | None, Query()] = None,
    intent: Annotated[str | None, Query()] = None,
) -> Response:
    records = await service.export_logs(
        session_id=session_id,
        parser_source=parser_source,
        agent_state=agent_state,
        intent=intent,
    )
    rows = [item.model_dump(mode="json") for item in records]
    return _csv_response(
        filename_prefix="admin_agent_history",
        rows=rows,
        fieldnames=[
            "id",
            "session_id",
            "user_input",
            "confirmed",
            "parser_source",
            "intent",
            "agent_state",
            "tool_name",
            "target_id",
            "user_message",
            "context",
            "tool_arguments",
            "missing_fields",
            "suggested_inputs",
            "execution_result",
            "created_at",
            "updated_at",
        ],
    )


@router.get(
    "/agent/sessions/{session_id}/history",  # 单个会话历史
    response_model=ApiResponse[list[ConversationHistoryRead]],
    status_code=status.HTTP_200_OK,
)
async def get_admin_agent_session_history(
    session_id: str,
    service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
) -> ApiResponse[list[ConversationHistoryRead]]:
    result = await service.list_by_session_id(session_id)
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/agent/sessions/{session_id}",  # 单个会话详情
    response_model=ApiResponse[ConversationSessionView],
    status_code=status.HTTP_200_OK,
)
async def get_admin_agent_session(
    session_id: str,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> ApiResponse[ConversationSessionView]:
    result = await service.get_session(session_id)
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/agent/llm-audit",  # LLM 审计日志
    response_model=ApiResponse[LLMAuditLogList],
    status_code=status.HTTP_200_OK)
async def list_admin_llm_audit_logs(
    service: Annotated[LLMAuditLogService, Depends(get_llm_audit_log_service)],
    session_id: Annotated[str | None, Query()] = None,
    parser_stage: Annotated[str | None, Query()] = None,
    success: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[LLMAuditLogList]:
    result = await service.list_logs(
        session_id=session_id,
        parser_stage=parser_stage,
        success=success,
        limit=limit,
        offset=offset,
    )
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/agent/llm-audit/export",  # 导出 LLM 审计
    response_class=Response,
    status_code=status.HTTP_200_OK)
async def export_admin_llm_audit_logs(
    service: Annotated[LLMAuditLogService, Depends(get_llm_audit_log_service)],
    session_id: Annotated[str | None, Query()] = None,
    parser_stage: Annotated[str | None, Query()] = None,
    success: Annotated[bool | None, Query()] = None,
) -> Response:
    records = await service.export_logs(
        session_id=session_id,
        parser_stage=parser_stage,
        success=success,
    )
    rows = [item.model_dump(mode="json") for item in records]
    return _csv_response(
        filename_prefix="admin_llm_audit",
        rows=rows,
        fieldnames=[
            "id",
            "session_id",
            "user_input",
            "parser_stage",
            "provider",
            "model_name",
            "success",
            "request_payload",
            "raw_response_text",
            "parsed_response",
            "error_message",
            "created_at",
            "updated_at",
        ],
    )


@router.get(
    "/agent/llm-audit/{session_id}",  # 单个会话的审计
    response_model=ApiResponse[list[LLMAuditLogRead]],
    status_code=status.HTTP_200_OK,
)
async def list_admin_llm_audit_logs_by_session(
    session_id: str,
    service: Annotated[LLMAuditLogService, Depends(get_llm_audit_log_service)],
) -> ApiResponse[list[LLMAuditLogRead]]:
    result = await service.list_by_session_id(session_id)
    return ApiResponse(code=200, message="success", data=result)

'''
和普通用户路由的区别：
增加了 导出 CSV 功能
可以查看所有用户的会话（普通用户只能看自己的）
'''

@router.get(
    "/access-audit",  # 访问审计日志
    response_model=ApiResponse[AdminAccessAuditLogList],
    status_code=status.HTTP_200_OK)
async def list_admin_access_audit_logs(
    service: Annotated[AdminAccessAuditService, Depends(get_admin_audit_service)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[AdminAccessAuditLogList]:
    result = await service.list_logs(limit=limit, offset=offset)
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/access-audit/export",   # 导出审计日志
    response_class=Response,
    status_code=status.HTTP_200_OK)
async def export_admin_access_audit_logs(
    service: Annotated[AdminAccessAuditService, Depends(get_admin_audit_service)],
) -> Response:
    records = await service.export_logs()
    rows = [item.model_dump(mode="json") for item in records]
    return _csv_response(
        filename_prefix="admin_access_audit",
        rows=rows,
        fieldnames=[
            "id",
            "path",
            "method",
            "client_host",
            "request_id",
            "access_granted",
            "auth_mode",
            "failure_reason",
            "created_at",
            "updated_at",
        ],
    )
