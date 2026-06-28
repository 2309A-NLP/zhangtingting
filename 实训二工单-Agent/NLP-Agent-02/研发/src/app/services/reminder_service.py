from datetime import datetime
# 负责“扫描哪些提醒到点了，并把它们放进待执行队列里”的。
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from app.core.enums import ReminderStatus
from app.core.exceptions import ApplicationError
from app.models.schedule import Schedule
from app.queue import ReminderQueueBackend, get_reminder_queue_backend
from app.repositories.schedule_repository import ScheduleRepository

logger = get_logger()
'''
ReminderService
├── 初始化（依赖注入）
│   ├── ScheduleRepository（数据访问层）
│   └── ReminderQueueBackend（消息队列后端）
├── 核心调度
│   └── scan_and_enqueue_due_reminders() → 扫描到期提醒并入队
├── 私有方法
│   └── _enqueue_due_schedule() → 处理单个到期日程（幂等性保护）
├── 幂等性检查
│   ├── 检查进行中日志（RETRYING / PENDING 状态）
│   ├── 检查已存在日志（所有状态）
│   └── 唯一约束冲突捕获（ApplicationError）
├── 日程查询
│   ├── list_due_active_schedules() → 查询到期活跃日程
│   └── advance_schedule_after_reminder() → 推进日程到下次触发
├── 提醒日志操作
│   ├── get_latest_reminder_log() → 获取最新日志
│   ├── get_existing_reminder_log() → 检查已存在日志
│   ├── create_reminder_log() → 创建提醒日志
│   └── list_retryable_reminder_logs() → 查询可重试日志
├── 投递任务操作
│   ├── create_delivery_task() → 创建投递任务
│   └── enqueue_task() → 任务入队
└── 可观测性
    ├── reminder_existing_inflight_log_detected → 进行中日志检测
    ├── reminder_skipped_duplicate → 重复跳过
    └── reminder_skipped_by_unique_constraint → 并发冲突跳过
'''

class ReminderService:
    def __init__(self, session: AsyncSession, queue_backend: ReminderQueueBackend | None = None) -> None:
        self._repository = ScheduleRepository(session)
        self._queue_backend = queue_backend or get_reminder_queue_backend()

    # 扫描现在该处理的提醒，然后加入队列，最后返回这次一共入队了多少条。
    async def scan_and_enqueue_due_reminders(self) -> int:
        now = datetime.now()
        # 正常到点该提醒的日程
        due_schedules = await self._repository.list_due_active_schedules(now)
        # 该重试发送的提醒日志
        retry_logs = await self._repository.list_retryable_reminder_logs(now)
        queued_count = 0

        for schedule in due_schedules:
            queued_count += await self._enqueue_due_schedule(schedule, now)

        for reminder_log in retry_logs:
            task = await self._repository.create_delivery_task(
                schedule_id=reminder_log.schedule_id,
                reminder_log_id=reminder_log.id,
                available_at=now,
            )
            await self._queue_backend.enqueue_task(task.id)
            queued_count += 1

        return queued_count

    async def _enqueue_due_schedule(self, schedule: Schedule, now: datetime) -> int:
        # 先看有没有触发时间
        planned_trigger_at = schedule.next_trigger_at
        if planned_trigger_at is None:
            return 0

        # 检查有没有“正在处理中的提醒日志”
        # 如果最新一条 reminder log 状态还是 PENDING 或 RETRYING
        # 说明“还在流程中”
        # 那就不要再重复入队
        latest_log = await self._repository.get_latest_reminder_log(
            schedule_id=schedule.id,
            planned_trigger_at=planned_trigger_at,
        )
        if latest_log is not None and latest_log.status in {
            ReminderStatus.RETRYING.value,
            ReminderStatus.PENDING.value,
        }:
            logger.info(
                "reminder_existing_inflight_log_detected",
                schedule_id=schedule.id,
                reminder_log_id=latest_log.id,
                status=latest_log.status,
            )
            return 0

        # 检查这次提醒是不是已经成功发过
        # 如果这次计划触发时间对应的提醒已经成功发过
        # 那就不要再发第二次
        # 而且还要把 schedule 往下一次推进
        existing_log = await self._repository.get_existing_reminder_log(
            schedule_id=schedule.id,
            planned_trigger_at=planned_trigger_at,
        )
        if existing_log is not None:
            logger.info(
                "reminder_skipped_duplicate",
                schedule_id=schedule.id,
                planned_trigger_at=str(planned_trigger_at),
            )
            await self._repository.advance_schedule_after_reminder(schedule)
            return 0

        # 如果既没在处理中，也没成功发过，就正式创建提醒任务
        try:
            reminder_log = await self._repository.create_reminder_log(
                schedule_id=schedule.id,
                planned_trigger_at=planned_trigger_at,
            )
            task = await self._repository.create_delivery_task(
                schedule_id=schedule.id,
                reminder_log_id=reminder_log.id,
                available_at=now,
            )
            await self._queue_backend.enqueue_task(task.id)
            return 1
        # 防止并发情况下的重复创建
        except ApplicationError:
            logger.info(
                "reminder_skipped_by_unique_constraint",
                schedule_id=schedule.id,
                planned_trigger_at=str(planned_trigger_at),
            )
            await self._repository.advance_schedule_after_reminder(schedule)
            return 0
