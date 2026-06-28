from datetime import datetime
from typing import Annotated
# 日程（Schedule）相关的业务逻辑
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db_session
from app.queue import ReminderQueueBackend, get_reminder_queue_backend
from app.repositories.schedule_repository import ScheduleRepository
from app.schemas.schedule import (
    ReminderAlertLogList,
    ReminderAlertLogQuery,
    ReminderAlertLogRead,
    ReminderDeliveryQueueSummary,
    ReminderDeliveryTaskList,
    ReminderDeliveryTaskQuery,
    ReminderDeliveryTaskRead,
    ReminderLogList,
    ReminderLogQuery,
    ReminderLogRead,
    ReminderReliabilitySummary,
    ScheduleCreate,
    ScheduleList,
    ScheduleQuery,
    ScheduleRead,
    ScheduleSummary,
    ScheduleUpdate,
)

'''
ScheduleService
├── 初始化（依赖注入）
│   ├── ScheduleRepository（数据访问层）
│   └── ReminderQueueBackend（消息队列）
├── 工厂方法
│   └── from_session() → 从数据库会话创建 Service
├── 日程管理（CRUD）
│   ├── create()    → 创建日程
│   ├── get_by_id() → 获取单个日程
│   ├── update()    → 更新日程
│   ├── delete()    → 删除日程
│   ├── list_all()  → 列表（分页 + 过滤）
│   ├── export_all()→ 导出全部
│   └── summarize() → 统计摘要
├── 提醒日志
│   ├── list_reminder_logs()    → 列表
│   ├── export_reminder_logs()  → 导出
│   ├── list_alert_logs()       → 告警日志
│   ├── export_alert_logs()     → 导出告警
│   └── summarize_reliability() → 可靠性统计
└── 投递任务管理
    ├── list_delivery_tasks()       → 列表
    ├── export_delivery_tasks()     → 导出
    ├── summarize_delivery_queue()  → 队列摘要
    ├── retry_delivery_task()       → 重试任务
    └── unlock_delivery_task()      → 解锁任务
'''
class ScheduleService:
    def __init__(
        self,
        repository: ScheduleRepository,
        queue_backend: ReminderQueueBackend | None = None,
    ) -> None:
        self._repository = repository
        self._queue_backend = queue_backend or get_reminder_queue_backend()

    @classmethod
    def from_session(cls, session: AsyncSession) -> "ScheduleService":
        return cls( # 调用__init__()方法
            ScheduleRepository(session),
            queue_backend=get_reminder_queue_backend(),
        )

    # 创建日程
    async def create(self, payload: ScheduleCreate) -> ScheduleRead:
        schedule = await self._repository.create(payload)
        # model_validate() 自动转换 + 验证
        return ScheduleRead.model_validate(schedule)

    # 列表查询（带分页 + 过滤）
    async def list_all(self, query: ScheduleQuery) -> ScheduleList:
        schedules, total = await self._repository.list_all(query)
        return ScheduleList(
            total=total,
            limit=query.limit,
            offset=query.offset,
            items=[ScheduleRead.model_validate(item) for item in schedules],
        )

    async def export_all(self, query: ScheduleQuery) -> list[ScheduleRead]:
        schedules = await self._repository.export_all(query)
        return [ScheduleRead.model_validate(item) for item in schedules]

    async def summarize(self, query: ScheduleQuery) -> ScheduleSummary:
        summary = await self._repository.summarize(query, now=datetime.now())
        return ScheduleSummary(**summary)

    # 提醒日志
    async def list_reminder_logs(self, query: ReminderLogQuery) -> ReminderLogList:
        logs, total = await self._repository.list_reminder_logs(query)
        return ReminderLogList(
            total=total,
            limit=query.limit,
            offset=query.offset,
            items=[ReminderLogRead.model_validate(item) for item in logs],
        )

    async def export_reminder_logs(self, query: ReminderLogQuery) -> list[ReminderLogRead]:
        logs, _ = await self._repository.list_reminder_logs(
            ReminderLogQuery(
                schedule_id=query.schedule_id,
                status=query.status,
                limit=100000,
                offset=0,
            )
        )
        return [ReminderLogRead.model_validate(item) for item in logs]

    async def list_alert_logs(self, query: ReminderAlertLogQuery) -> ReminderAlertLogList:
        logs, total = await self._repository.list_alert_logs(query)
        return ReminderAlertLogList(
            total=total,
            limit=query.limit,
            offset=query.offset,
            items=[ReminderAlertLogRead.model_validate(item) for item in logs],
        )

    async def export_alert_logs(self, query: ReminderAlertLogQuery) -> list[ReminderAlertLogRead]:
        logs, _ = await self._repository.list_alert_logs(
            ReminderAlertLogQuery(
                schedule_id=query.schedule_id,
                status=query.status,
                alert_type=query.alert_type,
                limit=100000,
                offset=0,
            )
        )
        return [ReminderAlertLogRead.model_validate(item) for item in logs]

    async def summarize_reliability(self) -> ReminderReliabilitySummary:
        reminder_summary = await self._repository.summarize_reminder_logs()
        alert_summary = await self._repository.summarize_alert_logs()
        return ReminderReliabilitySummary(
            total_logs=reminder_summary["total"],
            pending_count=reminder_summary["pending_count"],
            retrying_count=reminder_summary["retrying_count"],
            sent_count=reminder_summary["sent_count"],
            failed_count=reminder_summary["failed_count"],
            max_attempt_reached_count=reminder_summary["max_attempt_reached_count"],
            alert_total=alert_summary["total"],
            alert_sent_count=alert_summary["sent_count"],
            alert_failed_count=alert_summary["failed_count"],
        )

    async def list_delivery_tasks(self, query: ReminderDeliveryTaskQuery) -> ReminderDeliveryTaskList:
        tasks, total = await self._repository.list_delivery_tasks(query)
        return ReminderDeliveryTaskList(
            total=total,
            limit=query.limit,
            offset=query.offset,
            items=[ReminderDeliveryTaskRead.model_validate(item) for item in tasks],
        )

    async def export_delivery_tasks(self, query: ReminderDeliveryTaskQuery) -> list[ReminderDeliveryTaskRead]:
        tasks = await self._repository.export_delivery_tasks(query)
        return [ReminderDeliveryTaskRead.model_validate(item) for item in tasks]

    # 队列摘要  作用： 查看投递队列的整体状态
    async def summarize_delivery_queue(self) -> ReminderDeliveryQueueSummary:
        summary = await self._repository.summarize_delivery_tasks(
            now=datetime.now(),
            worker_lock_timeout_seconds=settings.worker_lock_timeout_seconds,
        )
        summary["redis_enabled"] = self._queue_backend.enabled
        summary["redis_queue_backlog"] = await self._queue_backend.get_queue_length()
        return ReminderDeliveryQueueSummary(**summary)

    # 投递任务管理
    '''
    管理员点击"重试"
        ↓
    1. 获取投递任务
    2. 获取关联的提醒日志
    3. 重置提醒日志状态（允许重试）
    4. 重新把任务放入队列（available_at = 现在）
    5. 通过 Redis 队列下发任务
    6. 返回更新后的任务状态
    
    _queue_backend.enqueue_task() → 把任务放入 Redis 队列
    Worker 会从队列取任务执行
    '''
    async def retry_delivery_task(self, task_id: int) -> ReminderDeliveryTaskRead:
        task = await self._repository.get_delivery_task_by_id(task_id)
        reminder_log = await self._repository.get_reminder_log_by_id(task.reminder_log_id)
        await self._repository.reset_reminder_log_for_manual_retry(reminder_log)
        await self._repository.requeue_delivery_task(
            task,
            available_at=datetime.now(),
            error_message="Retried manually from admin endpoint",
        )
        await self._queue_backend.enqueue_task(task.id)
        refreshed = await self._repository.get_delivery_task_by_id(task_id)
        return ReminderDeliveryTaskRead.model_validate(refreshed)

    # 解锁投递任务
    async def unlock_delivery_task(self, task_id: int) -> ReminderDeliveryTaskRead:
        task = await self._repository.get_delivery_task_by_id(task_id)
        await self._repository.requeue_delivery_task(
            task,
            available_at=datetime.now(),
            error_message="Unlocked manually from admin endpoint",
        )
        await self._queue_backend.enqueue_task(task.id)
        refreshed = await self._repository.get_delivery_task_by_id(task_id)
        return ReminderDeliveryTaskRead.model_validate(refreshed)

    async def get_by_id(self, schedule_id: int) -> ScheduleRead:
        schedule = await self._repository.get_by_id(schedule_id)
        return ScheduleRead.model_validate(schedule)

    async def update(self, schedule_id: int, payload: ScheduleUpdate) -> ScheduleRead:
        schedule = await self._repository.update(schedule_id, payload)
        return ScheduleRead.model_validate(schedule)

    async def delete(self, schedule_id: int) -> None:
        await self._repository.delete(schedule_id)


async def get_schedule_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ScheduleService:
    return ScheduleService.from_session(session)
