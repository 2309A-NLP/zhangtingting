from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Select, case, func, or_, select
# 唯一约束冲突
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
# 列表达式类型
from sqlalchemy.sql.elements import ColumnElement

from app.core.enums import CycleRule, ReminderStatus, ScheduleStatus
from app.core.exceptions import ApplicationError, NotFoundError
from app.models.schedule import ReminderAlertLog, ReminderDeliveryTask, ReminderLog, Schedule
from app.schemas.schedule import (
    ReminderAlertLogQuery,
    ReminderDeliveryTaskQuery,
    ReminderLogQuery,
    ScheduleCreate,
    ScheduleQuery,
    ScheduleUpdate,
)
from app.utils.datetime import calculate_following_trigger_at, calculate_next_trigger_at

'''
ScheduleRepository
├── 初始化
│   └── AsyncSession（数据库会话）
├── 日程管理（Schedule）
│   ├── create() → 创建日程
│   ├── list_all() → 列表（分页+过滤）
│   ├── export_all() → 导出全部
│   ├── summarize() → 统计摘要
│   ├── get_by_id() → 获取单个
│   ├── update() → 更新日程
│   ├── delete() → 删除日程（软删除）
│   └── list_due_active_schedules() → 查询到期日程
├── 提醒日志（ReminderLog）
│   ├── create_reminder_log() → 创建日志
│   ├── get_latest_reminder_log() → 获取最新日志
│   ├── get_existing_reminder_log() → 检查已存在日志
│   ├── list_reminder_logs() → 列表
│   ├── summarize_reminder_logs() → 统计
│   ├── list_retryable_reminder_logs() → 查询可重试日志
│   ├── mark_reminder_sent() → 标记已发送
│   ├── mark_reminder_failed() → 标记失败
│   ├── mark_reminder_retrying() → 标记重试中
│   └── reset_reminder_log_for_manual_retry() → 重置重试
├── 投递任务（ReminderDeliveryTask）
│   ├── create_delivery_task() → 创建任务
│   ├── get_delivery_task_by_id() → 获取单个
│   ├── list_delivery_tasks() → 列表
│   ├── export_delivery_tasks() → 导出
│   ├── claim_delivery_tasks() → 领取任务（Worker）
│   ├── claim_delivery_tasks_by_ids() → 按ID领取
│   ├── requeue_stale_processing_tasks() → 重新入队僵尸任务
│   ├── mark_delivery_task_done() → 标记完成
│   ├── mark_delivery_task_failed() → 标记失败
│   ├── requeue_delivery_task() → 重新入队
│   └── summarize_delivery_tasks() → 统计
├── 告警日志（ReminderAlertLog）
│   ├── create_alert_log() → 创建告警
│   ├── list_alert_logs() → 列表
│   ├── summarize_alert_logs() → 统计
│   ├── get_latest_alert_by_dedupe_key() → 去重检查
│   ├── mark_alert_sent() → 标记已发送
│   └── mark_alert_failed() → 标记失败
├── 日程推进
│   └── advance_schedule_after_reminder() → 推进到下次触发
└── 指标快照
    └── build_metrics_snapshot() → 构建 Prometheus 指标
'''

class ScheduleRepository:
    # 排序字段映射
    _SORT_FIELD_MAP = {
        "id": Schedule.id,
        "created_at": Schedule.created_at,
        "updated_at": Schedule.updated_at,
        "schedule_date": Schedule.schedule_date,
        "schedule_time": Schedule.schedule_time,
        "next_trigger_at": Schedule.next_trigger_at,
    }

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # 循环规则类型转换
    @staticmethod
    def _coerce_cycle_rule(value: CycleRule | str) -> CycleRule:
        if isinstance(value, CycleRule):
            return value
        return CycleRule(value)

    # 应用查询过滤
    # 根据查询条件动态构建 SQL 查询语句。
    @staticmethod
    def _apply_query_filters(statement: Select[Any], query: ScheduleQuery) -> Select[Any]:
        if query.date is not None:
            statement = statement.where(Schedule.schedule_date == query.date)
        if query.start_date is not None:
            statement = statement.where(Schedule.schedule_date >= query.start_date)
        if query.end_date is not None:
            statement = statement.where(Schedule.schedule_date <= query.end_date)
        if query.status is not None:
            statement = statement.where(Schedule.status == query.status.value)
        if query.statuses:
            statement = statement.where(Schedule.status.in_([item.value for item in query.statuses]))
        if query.schedule_time_start is not None:
            statement = statement.where(Schedule.schedule_time >= query.schedule_time_start)
        if query.schedule_time_end is not None:
            statement = statement.where(Schedule.schedule_time <= query.schedule_time_end)
        if query.keyword is not None:
            keyword = f"%{query.keyword}%"
            statement = statement.where(
                or_(
                    Schedule.content.ilike(keyword),
                    Schedule.source_text.ilike(keyword),
                )
            )
        return statement

    # 构建排序
    @staticmethod
    def _build_order_by(query: ScheduleQuery) -> ColumnElement[Any]:
        sort_column = ScheduleRepository._SORT_FIELD_MAP[query.sort_by]
        return sort_column.asc() if query.sort_order == "asc" else sort_column.desc()

    # 创建日程
    async def create(self, payload: ScheduleCreate) -> Schedule:
        # 将 Pydantic 对象转换为字典
        payload_data = payload.model_dump()
        payload_data["cycle_rule"] = payload.cycle_rule.value
        next_trigger_at = calculate_next_trigger_at(
            schedule_date=payload.schedule_date,
            schedule_time=payload.schedule_time,
            cycle_rule=payload.cycle_rule,
            cycle_value=payload.cycle_value,
        )
        schedule = Schedule(**payload_data, next_trigger_at=next_trigger_at)
        self._session.add(schedule)
        await self._session.commit()
        await self._session.refresh(schedule)
        return schedule

    # 列表查询
    async def list_all(self, query: ScheduleQuery) -> tuple[list[Schedule], int]:
        statement: Select[tuple[Schedule]] = select(Schedule)
        count_statement = select(func.count()).select_from(Schedule)
        statement = self._apply_query_filters(statement, query)
        count_statement = self._apply_query_filters(count_statement, query)
        statement = statement.order_by(
            self._build_order_by(query),
            Schedule.id.desc(),
        ).offset(query.offset).limit(query.limit)

        result = await self._session.execute(statement)
        count_result = await self._session.execute(count_statement)
        # result.scalars()	提取标量值（模型对象）
        return list(result.scalars().all()), int(count_result.scalar_one())

    # 导出
    async def export_all(self, query: ScheduleQuery) -> list[Schedule]:
        statement: Select[tuple[Schedule]] = select(Schedule)
        statement = self._apply_query_filters(statement, query)
        statement = statement.order_by(
            self._build_order_by(query),
            Schedule.id.desc(),
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    # 统计摘要
    async def summarize(self, query: ScheduleQuery, now: datetime) -> dict[str, int]:
        today = now.date()
        base_statement: Select[tuple[Schedule]] = select(Schedule)
        # .subquery()	转换为子查询，供其他统计语句复用
        filtered_subquery = self._apply_query_filters(base_statement, query).subquery()

        # 1.总数统计
        total_statement = select(func.count()).select_from(filtered_subquery)
        # 2.按状态分组统计
        # .c 是 SQLAlchemy 中访问子查询/表列（column）的属性。
        status_statement = select(
            filtered_subquery.c.status,
            func.count().label("count"),
        ).group_by(filtered_subquery.c.status)
        # 3.今日到期统计
        due_today_statement = select(func.count()).select_from(filtered_subquery).where(
            filtered_subquery.c.status == ScheduleStatus.ACTIVE.value,
            func.date(filtered_subquery.c.next_trigger_at) == today.isoformat(),
        )
        # 4.已逾期统计
        overdue_statement = select(func.count()).select_from(filtered_subquery).where(
            filtered_subquery.c.status == ScheduleStatus.ACTIVE.value,
            filtered_subquery.c.next_trigger_at.is_not(None),
            filtered_subquery.c.next_trigger_at < now,
        )
        # 5.即将到来统计
        upcoming_statement = select(func.count()).select_from(filtered_subquery).where(
            filtered_subquery.c.status == ScheduleStatus.ACTIVE.value,
            filtered_subquery.c.next_trigger_at.is_not(None),
            filtered_subquery.c.next_trigger_at >= now,
        )

        total_result = await self._session.execute(total_statement)
        status_result = await self._session.execute(status_statement)
        due_today_result = await self._session.execute(due_today_statement)
        overdue_result = await self._session.execute(overdue_statement)
        upcoming_result = await self._session.execute(upcoming_statement)

        status_counts = {str(row[0]): int(row[1]) for row in status_result}
        # scalar_one()  是 SQLAlchemy 的方法，用于执行查询并返回单个值，如果结果不是一条记录则报错。
        return {
            "total": int(total_result.scalar_one()),
            "active_count": status_counts.get(ScheduleStatus.ACTIVE.value, 0),
            "cancelled_count": status_counts.get(ScheduleStatus.CANCELLED.value, 0),
            "done_count": status_counts.get(ScheduleStatus.DONE.value, 0),
            "due_today_count": int(due_today_result.scalar_one()),
            "overdue_count": int(overdue_result.scalar_one()),
            "upcoming_count": int(upcoming_result.scalar_one()),
        }

    # 按id查找日程
    async def get_by_id(self, schedule_id: int) -> Schedule:
        result = await self._session.execute(select(Schedule).where(Schedule.id == schedule_id))
        schedule = result.scalar_one_or_none()
        if schedule is None:
            raise NotFoundError(f"Schedule {schedule_id} not found")
        return schedule

    # 更新现有的日程记录
    async def update(self, schedule_id: int, payload: ScheduleUpdate) -> Schedule:
        schedule = await self.get_by_id(schedule_id)
        # 场景	                   行为	                        示例
        # exclude_unset=False	包含所有字段（包括未设置的）	   {"content": None, "date": None, "time": None}
        # exclude_unset=True	只包含明确设置的字段	       {"content": "新内容"}
        update_data = payload.model_dump(exclude_unset=True)
        for field_name, value in update_data.items():
            # 批量更新字段
            setattr(schedule, field_name, value.value if hasattr(value, "value") else value)

        schedule.next_trigger_at = calculate_next_trigger_at(
            schedule_date=schedule.schedule_date,
            schedule_time=schedule.schedule_time,
            cycle_rule=self._coerce_cycle_rule(schedule.cycle_rule),
            cycle_value=schedule.cycle_value,
        )

        await self._session.commit()
        await self._session.refresh(schedule)
        return schedule

    # 删除   将指定日程的状态标记为 CANCELLED，并清空下次触发时间，而不是真正从数据库中删除记录。
    async def delete(self, schedule_id: int) -> None:
        schedule = await self.get_by_id(schedule_id)
        schedule.status = ScheduleStatus.CANCELLED.value
        schedule.next_trigger_at = None
        await self._session.commit()

    # 查询所有"应该触发但还未触发的活跃日程"
    async def list_due_active_schedules(self, now: datetime) -> list[Schedule]:
        statement = (
            select(Schedule)
            .where(Schedule.status == ScheduleStatus.ACTIVE.value)
            .where(Schedule.next_trigger_at.is_not(None))
            .where(Schedule.next_trigger_at <= now)
            .order_by(Schedule.next_trigger_at.asc())
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    # 查询是否已经存在特定日程在特定时间点的已发送提醒记录，用于防止重复发送
    async def get_existing_reminder_log(
        self, schedule_id: int, planned_trigger_at: datetime
    ) -> ReminderLog | None:
        statement = select(ReminderLog).where(
            ReminderLog.schedule_id == schedule_id,
            ReminderLog.planned_trigger_at == planned_trigger_at,
            ReminderLog.status == ReminderStatus.SENT.value,
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    # 查询某个日程在指定触发时间点的所有提醒记录，按 ID 降序排序，返回最新的一条。
    async def get_latest_reminder_log(
        self, schedule_id: int, planned_trigger_at: datetime
    ) -> ReminderLog | None:
        statement = (
            select(ReminderLog)
            .where(
                ReminderLog.schedule_id == schedule_id,
                ReminderLog.planned_trigger_at == planned_trigger_at,
            )
            .order_by(ReminderLog.id.desc())
        )
        result = await self._session.execute(statement)
        return result.scalars().first()

    # 在发送提醒之前，先创建一条状态为 PENDING 的日志记录，用于追踪提醒的发送状态和重试次数
    async def create_reminder_log(
        self, schedule_id: int, planned_trigger_at: datetime
    ) -> ReminderLog:
        reminder_log = ReminderLog(
            schedule_id=schedule_id,
            planned_trigger_at=planned_trigger_at,
            status=ReminderStatus.PENDING.value,
            attempt_count=1,
        )
        self._session.add(reminder_log)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise ApplicationError("Duplicate reminder log detected") from exc
        return reminder_log

    # 在提醒成功发送后，更新日志记录的状态为 SENT，记录发送时间，并清空下次重试时间。
    async def mark_reminder_sent(self, reminder_log: ReminderLog, reminded_at: datetime) -> None:
        reminder_log.status = ReminderStatus.SENT.value
        reminder_log.reminded_at = reminded_at
        reminder_log.last_attempt_at = reminded_at
        reminder_log.next_retry_at = None
        await self._session.commit()

    # 在提醒发送失败后，更新日志记录的状态为 FAILED，记录错误信息，并清空下次重试时间。
    async def mark_reminder_failed(self, reminder_log: ReminderLog, error_message: str) -> None:
        reminder_log.status = ReminderStatus.FAILED.value
        reminder_log.error_message = error_message
        reminder_log.last_attempt_at = datetime.now()
        reminder_log.next_retry_at = None
        await self._session.commit()

    # 在提醒发送失败后，将状态更新为 RETRYING（重试中），记录错误信息，增加尝试次数，并设置下次重试时间。
    async def mark_reminder_retrying(
        self,
        reminder_log: ReminderLog,
        *,
        error_message: str,
        next_retry_at: datetime,
        attempted_at: datetime,
    ) -> None:
        reminder_log.status = ReminderStatus.RETRYING.value
        reminder_log.error_message = error_message
        reminder_log.last_attempt_at = attempted_at
        reminder_log.next_retry_at = next_retry_at
        reminder_log.attempt_count += 1
        await self._session.commit()

    # 找出所有状态为 RETRYING、有重试时间、且重试时间已到（或已过）的记录，交给调度器执行重试。
    async def list_retryable_reminder_logs(self, now: datetime) -> list[ReminderLog]:
        statement = (
            select(ReminderLog)
            .where(ReminderLog.status == ReminderStatus.RETRYING.value)
            .where(ReminderLog.next_retry_at.is_not(None))
            .where(ReminderLog.next_retry_at <= now)
            .order_by(ReminderLog.next_retry_at.asc(), ReminderLog.id.asc())
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    # 根据 ReminderLog 中的 schedule_id，查询并返回完整的 Schedule 对象。
    async def get_schedule_by_reminder_log(self, reminder_log: ReminderLog) -> Schedule:
        return await self.get_by_id(reminder_log.schedule_id)

    # 当系统需要发送告警（如通知、提醒、警告）时，创建一条状态为 pending 的告警日志，用于追踪告警的发送状态。
    async def create_alert_log(
        self,
        *,
        schedule_id: int,
        reminder_log_id: int | None,
        alert_type: str,
        alert_channel: str,
        message: str,
        dedupe_key: str | None = None,
    ) -> ReminderAlertLog:
        alert_log = ReminderAlertLog(
            schedule_id=schedule_id,
            reminder_log_id=reminder_log_id,
            alert_type=alert_type,
            alert_channel=alert_channel,
            status="pending",
            message=message,
            dedupe_key=dedupe_key,
        )
        self._session.add(alert_log)
        await self._session.commit()
        await self._session.refresh(alert_log)
        return alert_log

    # 当需要发送提醒时，创建一个投递任务（ReminderDeliveryTask），将发送工作放入任务队列，由后台 worker 异步处理。
    async def create_delivery_task(
        self,
        *,
        schedule_id: int,
        reminder_log_id: int,
        available_at: datetime,
        task_type: str = "send",
    ) -> ReminderDeliveryTask:
        task = ReminderDeliveryTask(
            schedule_id=schedule_id,
            reminder_log_id=reminder_log_id,
            task_type=task_type,
            status="queued",
            available_at=available_at,
        )
        self._session.add(task)
        await self._session.commit()
        await self._session.refresh(task)
        return task

    # 不需要指定任务 ID，自动查询所有 queued 且已到执行时间的任务，按时间顺序认领指定数量，锁定供当前 worker 处理。
    async def claim_delivery_tasks(
        self,
        *,
        now: datetime,
        worker_id: str,
        limit: int,
    ) -> list[ReminderDeliveryTask]:
        statement = (
            select(ReminderDeliveryTask)
            .where(ReminderDeliveryTask.status == "queued")
            .where(ReminderDeliveryTask.available_at <= now)
            .order_by(ReminderDeliveryTask.available_at.asc(), ReminderDeliveryTask.id.asc())
            .limit(limit)
        )
        result = await self._session.execute(statement)
        tasks = list(result.scalars().all())
        claimed: list[ReminderDeliveryTask] = []
        for task in tasks:
            task.status = "processing"
            task.locked_by = worker_id
            task.locked_at = now
            claimed.append(task)
        if claimed:
            await self._session.commit()
        return claimed

    # 根据任务 ID 列表查询可用的任务，将它们标记为 processing（处理中），并绑定到当前 worker，防止其他 worker 重复处理。
    async def claim_delivery_tasks_by_ids(
        self,
        *,
        task_ids: list[int],
        now: datetime,
        worker_id: str,
    ) -> list[ReminderDeliveryTask]:
        if not task_ids:
            return []
        statement = (
            select(ReminderDeliveryTask)
            .where(ReminderDeliveryTask.id.in_(task_ids))
            .where(ReminderDeliveryTask.status == "queued")
            .where(ReminderDeliveryTask.available_at <= now)
            .order_by(ReminderDeliveryTask.id.asc())
        )
        result = await self._session.execute(statement)
        tasks = list(result.scalars().all())
        for task in tasks:
            task.status = "processing"
            task.locked_by = worker_id
            task.locked_at = now
        if tasks:
            await self._session.commit()
        return tasks

    # 根据查询条件（日程 ID、状态）过滤投递任务，支持分页和排序，同时返回当前页数据和总记录数。
    async def list_delivery_tasks(
        self,
        query: ReminderDeliveryTaskQuery,
    ) -> tuple[list[ReminderDeliveryTask], int]:
        statement: Select[tuple[ReminderDeliveryTask]] = select(ReminderDeliveryTask)
        count_statement = select(func.count()).select_from(ReminderDeliveryTask)

        if query.schedule_id is not None:
            statement = statement.where(ReminderDeliveryTask.schedule_id == query.schedule_id)
            count_statement = count_statement.where(ReminderDeliveryTask.schedule_id == query.schedule_id)
        if query.status is not None:
            statement = statement.where(ReminderDeliveryTask.status == query.status)
            count_statement = count_statement.where(ReminderDeliveryTask.status == query.status)

        statement = statement.order_by(ReminderDeliveryTask.id.desc()).offset(query.offset).limit(query.limit)
        result = await self._session.execute(statement)
        count_result = await self._session.execute(count_statement)
        return list(result.scalars().all()), int(count_result.scalar_one())

    # 根据任务 ID 查询投递任务，如果存在则返回，不存在则抛出 NotFoundError
    async def get_delivery_task_by_id(self, task_id: int) -> ReminderDeliveryTask:
        result = await self._session.execute(
            select(ReminderDeliveryTask).where(ReminderDeliveryTask.id == task_id)
        )
        task = result.scalar_one_or_none()
        if task is None:
            raise NotFoundError(f"Delivery task {task_id} not found")
        return task

    # 根据查询条件过滤投递任务，返回所有符合条件的任务列表，不进行分页，用于数据导出或批量处理。
    async def export_delivery_tasks(self, query: ReminderDeliveryTaskQuery) -> list[ReminderDeliveryTask]:
        statement: Select[tuple[ReminderDeliveryTask]] = select(ReminderDeliveryTask)

        if query.schedule_id is not None:
            statement = statement.where(ReminderDeliveryTask.schedule_id == query.schedule_id)
        if query.status is not None:
            statement = statement.where(ReminderDeliveryTask.status == query.status)

        statement = statement.order_by(ReminderDeliveryTask.id.desc())
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    # 检测哪些任务在 processing 状态停留太久（被锁定的时间超过阈值），将它们重置为 queued 状态，以便其他 worker 可以重新认领和处理。
    async def requeue_stale_processing_tasks(
        self,
        *,
        stale_before: datetime,  # 超时阈值时间
        available_at: datetime,  # 重新入队后的可用时间
    ) -> int:
        statement = (
            select(ReminderDeliveryTask)
            .where(ReminderDeliveryTask.status == "processing")
            .where(ReminderDeliveryTask.locked_at.is_not(None))
            .where(ReminderDeliveryTask.locked_at <= stale_before)
            .order_by(ReminderDeliveryTask.locked_at.asc(), ReminderDeliveryTask.id.asc())
        )
        result = await self._session.execute(statement)
        tasks = list(result.scalars().all())
        for task in tasks:
            task.status = "queued"
            task.available_at = available_at
            task.locked_by = None
            task.locked_at = None
            task.last_error_message = "Recovered stale processing task"
        if tasks:
            await self._session.commit()
        return len(tasks)

    # 任务成功执行完毕后，更新状态为 done，释放 worker 绑定，清空锁定信息和错误信息。
    async def mark_delivery_task_done(self, task: ReminderDeliveryTask) -> None:
        task.status = "done"
        task.locked_by = None
        task.locked_at = None
        task.last_error_message = None
        await self._session.commit()

    # 任务执行失败后，更新状态为 failed，释放 worker 绑定，并记录错误信息，便于后续排查和重试
    async def mark_delivery_task_failed(self, task: ReminderDeliveryTask, error_message: str) -> None:
        task.status = "failed"
        task.locked_by = None
        task.locked_at = None
        task.last_error_message = error_message
        await self._session.commit()

    # 当任务执行失败但希望稍后重试时，将任务状态重置为 queued，更新可用时间，并记录错误信息，等待 worker 重新认领。
    async def requeue_delivery_task(self, task: ReminderDeliveryTask, *, available_at: datetime, error_message: str) -> None:
        task.status = "queued"
        task.available_at = available_at
        task.locked_by = None
        task.locked_at = None
        task.last_error_message = error_message
        await self._session.commit()

    # 当管理员或运维人员需要手动重新发送某个提醒时，将该提醒的日志重置为初始状态（PENDING），清除错误信息和重试时间，让它像新任务一样被重新处理。
    async def reset_reminder_log_for_manual_retry(self, reminder_log: ReminderLog) -> None:
        reminder_log.status = ReminderStatus.PENDING.value
        reminder_log.error_message = None
        reminder_log.next_retry_at = None
        await self._session.commit()

    # 根据提醒日志 ID 查询记录，如果存在则返回 ReminderLog 对象，不存在则抛出 NotFoundError。
    async def get_reminder_log_by_id(self, reminder_log_id: int) -> ReminderLog:
        result = await self._session.execute(select(ReminderLog).where(ReminderLog.id == reminder_log_id))
        reminder_log = result.scalar_one_or_none()
        if reminder_log is None:
            raise NotFoundError(f"Reminder log {reminder_log_id} not found")
        return reminder_log

    # 使用 dedupe_key 查询告警日志，返回最新的一条（按 ID 降序），用于检查某个告警是否已经发送过，避免重复发送。
    async def get_latest_alert_by_dedupe_key(self, dedupe_key: str) -> ReminderAlertLog | None:
        statement = (
            select(ReminderAlertLog)
            .where(ReminderAlertLog.dedupe_key == dedupe_key)
            .order_by(ReminderAlertLog.id.desc())
        )
        result = await self._session.execute(statement)
        return result.scalars().first()

    # 告警成功发送后，更新日志状态为 sent，记录实际发送时间，并清空错误信息
    async def mark_alert_sent(self, alert_log: ReminderAlertLog, sent_at: datetime) -> None:
        alert_log.status = "sent"
        alert_log.sent_at = sent_at
        alert_log.error_message = None
        await self._session.commit()

    # 告警发送失败后，更新日志状态为 failed，并记录错误信息，便于后续排查和重试。
    async def mark_alert_failed(self, alert_log: ReminderAlertLog, error_message: str) -> None:
        alert_log.status = "failed"
        alert_log.error_message = error_message
        await self._session.commit()

    # 当日程的提醒被成功发送后，根据循环规则计算下一次触发时间，如果是单次日程则标记为 DONE。
    async def advance_schedule_after_reminder(self, schedule: Schedule) -> None:
        if schedule.next_trigger_at is None:
            return

        next_trigger_at = calculate_following_trigger_at(
            current_trigger_at=schedule.next_trigger_at,
            schedule_time=schedule.schedule_time,
            cycle_rule=self._coerce_cycle_rule(schedule.cycle_rule),
            cycle_value=schedule.cycle_value,
        )

        if next_trigger_at is None:
            schedule.status = ScheduleStatus.DONE.value
            schedule.next_trigger_at = None
        else:
            schedule.next_trigger_at = next_trigger_at

        await self._session.commit()

    # 一次性统计所有提醒日志的总数及各状态分布，用于监控看板或数据大盘。
    async def summarize_reminder_logs(self) -> dict[str, int]:
        statement = select(
            func.count().label("total"),
            func.coalesce( # coalesce作用： 如果结果为 NULL，返回 0（防止字典中出现 None）
                func.sum(case((ReminderLog.status == ReminderStatus.PENDING.value, 1), else_=0)),
                0,
            ).label("pending_count"),
            func.coalesce(
                func.sum(case((ReminderLog.status == ReminderStatus.RETRYING.value, 1), else_=0)),
                0,
            ).label("retrying_count"),
            func.coalesce(
                func.sum(case((ReminderLog.status == ReminderStatus.SENT.value, 1), else_=0)),
                0,
            ).label("sent_count"),
            func.coalesce(
                func.sum(case((ReminderLog.status == ReminderStatus.FAILED.value, 1), else_=0)),
                0,
            ).label("failed_count"),
            func.coalesce(
                func.sum(case((ReminderLog.status == ReminderStatus.FAILED.value, 1), else_=0)),
                0,
            ).label("max_attempt_reached_count"),
        ).select_from(ReminderLog)

        result = await self._session.execute(statement)
        row = result.one()
        return {
            "total": int(row.total),
            "pending_count": int(row.pending_count),
            "retrying_count": int(row.retrying_count),
            "sent_count": int(row.sent_count),
            "failed_count": int(row.failed_count),
            "max_attempt_reached_count": int(row.max_attempt_reached_count),
        }

    # 根据查询条件（日程 ID、状态）过滤提醒日志，支持分页和排序，同时返回当前页数据和总记录数。
    async def list_reminder_logs(self, query: ReminderLogQuery) -> tuple[list[ReminderLog], int]:
        statement: Select[tuple[ReminderLog]] = select(ReminderLog)
        count_statement = select(func.count()).select_from(ReminderLog)

        if query.schedule_id is not None:
            statement = statement.where(ReminderLog.schedule_id == query.schedule_id)
            count_statement = count_statement.where(ReminderLog.schedule_id == query.schedule_id)
        if query.status is not None:
            statement = statement.where(ReminderLog.status == query.status.value)
            count_statement = count_statement.where(ReminderLog.status == query.status.value)

        statement = (
            statement.order_by(ReminderLog.id.desc())
            .offset(query.offset)
            .limit(query.limit)
        )

        result = await self._session.execute(statement)
        count_result = await self._session.execute(count_statement)
        return list(result.scalars().all()), int(count_result.scalar_one())

    # 根据查询条件（日程 ID、状态、告警类型）过滤告警日志，支持分页和排序，同时返回当前页数据和总记录数。
    async def list_alert_logs(self, query: ReminderAlertLogQuery) -> tuple[list[ReminderAlertLog], int]:
        statement: Select[tuple[ReminderAlertLog]] = select(ReminderAlertLog)
        count_statement = select(func.count()).select_from(ReminderAlertLog)

        if query.schedule_id is not None:
            statement = statement.where(ReminderAlertLog.schedule_id == query.schedule_id)
            count_statement = count_statement.where(ReminderAlertLog.schedule_id == query.schedule_id)
        if query.status is not None:
            statement = statement.where(ReminderAlertLog.status == query.status)
            count_statement = count_statement.where(ReminderAlertLog.status == query.status)
        if query.alert_type is not None:
            statement = statement.where(ReminderAlertLog.alert_type == query.alert_type)
            count_statement = count_statement.where(ReminderAlertLog.alert_type == query.alert_type)

        statement = statement.order_by(ReminderAlertLog.id.desc()).offset(query.offset).limit(query.limit)
        result = await self._session.execute(statement)
        count_result = await self._session.execute(count_statement)
        return list(result.scalars().all()), int(count_result.scalar_one())

    # 一次性统计所有告警日志的总数、成功发送数、失败数，用于监控告警系统的健康状况。
    async def summarize_alert_logs(self) -> dict[str, int]:
        statement = select(
            func.count().label("total"),
            func.coalesce(
                func.sum(case((ReminderAlertLog.status == "sent", 1), else_=0)),
                0,
            ).label("sent_count"),
            func.coalesce(
                func.sum(case((ReminderAlertLog.status == "failed", 1), else_=0)),
                0,
            ).label("failed_count"),
        ).select_from(ReminderAlertLog)
        result = await self._session.execute(statement)
        row = result.one()
        return {
            "total": int(row.total),
            "sent_count": int(row.sent_count),
            "failed_count": int(row.failed_count),
        }

    # 一次性统计所有投递任务的总数及按状态分布，特别增加了"卡住的 processing 任务"统计，用于监控任务队列的健康状况
    async def summarize_delivery_tasks(
        self,
        *,
        now: datetime | None = None,
        worker_lock_timeout_seconds: int | None = None,
    ) -> dict[str, int]:
        # 计算 Stale 阈值
        stale_before = None
        if now is not None and worker_lock_timeout_seconds is not None:
            stale_before = now - timedelta(seconds=max(worker_lock_timeout_seconds, 1))

        statement = select(
            func.count().label("total"),
            func.coalesce(
                func.sum(case((ReminderDeliveryTask.status == "queued", 1), else_=0)),
                0,
            ).label("queued_count"),
            func.coalesce(
                func.sum(case((ReminderDeliveryTask.status == "processing", 1), else_=0)),
                0,
            ).label("processing_count"),
            func.coalesce(
                func.sum(
                    case(
                        (
                            (ReminderDeliveryTask.status == "processing")
                            & ReminderDeliveryTask.locked_at.is_not(None)
                            & (
                                ReminderDeliveryTask.locked_at
                                <= (stale_before if stale_before is not None else datetime.max)
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ),
                0,
            ).label("stale_processing_count"),
            func.coalesce(
                func.sum(case((ReminderDeliveryTask.status == "done", 1), else_=0)),
                0,
            ).label("done_count"),
            func.coalesce(
                func.sum(case((ReminderDeliveryTask.status == "failed", 1), else_=0)),
                0,
            ).label("failed_count"),
        ).select_from(ReminderDeliveryTask)
        result = await self._session.execute(statement)
        row = result.one()
        return {
            "total": int(row.total),
            "queued_count": int(row.queued_count),
            "processing_count": int(row.processing_count),
            "stale_processing_count": int(row.stale_processing_count),
            "done_count": int(row.done_count),
            "failed_count": int(row.failed_count),
        }

    # 一次性收集所有核心模块的统计数据，生成一个完整的系统健康状态快照，用于监控看板或健康检查。
    async def build_metrics_snapshot(self) -> dict[str, int]:
        reminder_summary = await self.summarize_reminder_logs()
        alert_summary = await self.summarize_alert_logs()
        delivery_task_summary = await self.summarize_delivery_tasks()
        active_schedule_statement = select(func.count()).select_from(Schedule).where(
            Schedule.status == ScheduleStatus.ACTIVE.value
        )
        active_schedule_result = await self._session.execute(active_schedule_statement)
        return {
            "schedule_active_total": int(active_schedule_result.scalar_one()),
            "reminder_total": reminder_summary["total"],
            "reminder_pending_total": reminder_summary["pending_count"],
            "reminder_retrying_total": reminder_summary["retrying_count"],
            "reminder_sent_total": reminder_summary["sent_count"],
            "reminder_failed_total": reminder_summary["failed_count"],
            "reminder_failed_max_attempt_total": reminder_summary["max_attempt_reached_count"],
            "reminder_alert_total": alert_summary["total"],
            "reminder_alert_sent_total": alert_summary["sent_count"],
            "reminder_alert_failed_total": alert_summary["failed_count"],
            "delivery_task_total": delivery_task_summary["total"],
            "delivery_task_queued_total": delivery_task_summary["queued_count"],
            "delivery_task_processing_total": delivery_task_summary["processing_count"],
            "delivery_task_stale_processing_total": delivery_task_summary["stale_processing_count"],
            "delivery_task_done_total": delivery_task_summary["done_count"],
            "delivery_task_failed_total": delivery_task_summary["failed_count"],
        }
