from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from app.channels.alert_base import ReminderAlertChannel
from app.channels.base import ReminderChannel
from app.channels.factory import get_reminder_alert_channel, get_reminder_channel
from app.core.config import settings
from app.models.schedule import ReminderDeliveryTask, ReminderLog, Schedule
from app.queue import ReminderQueueBackend, get_reminder_queue_backend
from app.repositories.schedule_repository import ScheduleRepository

logger = get_logger()


class ReminderWorkerService:
    def __init__(
        self,
        session: AsyncSession,
        channel: ReminderChannel | None = None,
        alert_channel: ReminderAlertChannel | None = None,
        queue_backend: ReminderQueueBackend | None = None,
    ) -> None:
        self._repository = ScheduleRepository(session)
        self._channel = channel or get_reminder_channel()
        self._alert_channel = alert_channel or get_reminder_alert_channel()
        self._queue_backend = queue_backend or get_reminder_queue_backend()

    async def process_delivery_tasks(self) -> int:
        now = datetime.now()
        await self.recover_stale_tasks(now=now)
        tasks = await self._claim_tasks(now)
        processed_count = 0
        for task in tasks:
            await self._process_task(task, now)
            processed_count += 1
        return processed_count

    async def _claim_tasks(self, now: datetime) -> list[ReminderDeliveryTask]:
        redis_task_ids = await self._queue_backend.dequeue_task_ids(max_items=settings.worker_batch_size)
        if redis_task_ids:
            claimed = await self._repository.claim_delivery_tasks_by_ids(
                task_ids=redis_task_ids,
                now=now,
                worker_id=settings.worker_owner,
            )
            if claimed:
                return claimed
        return await self._repository.claim_delivery_tasks(
            now=now,
            worker_id=settings.worker_owner,
            limit=settings.worker_batch_size,
        )

    async def _process_task(self, task: ReminderDeliveryTask, now: datetime) -> None:
        schedule = await self._repository.get_by_id(task.schedule_id)
        reminder_log = await self._repository.get_reminder_log_by_id(task.reminder_log_id)
        try:
            await self._channel.send(schedule)
            await self._repository.mark_reminder_sent(reminder_log, now)
            await self._repository.mark_delivery_task_done(task)
            await self._repository.advance_schedule_after_reminder(schedule)
        except Exception as exc:  # pragma: no cover
            await self._handle_send_failure(schedule, reminder_log, task, str(exc), now)
            logger.exception("reminder_worker_send_failed", schedule_id=schedule.id, task_id=task.id)

    async def recover_stale_tasks(self, *, now: datetime | None = None) -> int:
        reference_time = now or datetime.now()
        stale_before = reference_time - timedelta(seconds=max(settings.worker_lock_timeout_seconds, 1))
        recovered_count = await self._repository.requeue_stale_processing_tasks(
            stale_before=stale_before,
            available_at=reference_time,
        )
        if recovered_count:
            logger.warning(
                "reminder_worker_recovered_stale_tasks",
                recovered_count=recovered_count,
                stale_before=stale_before.isoformat(),
            )
        return recovered_count

    async def _handle_send_failure(
        self,
        schedule: Schedule,
        reminder_log: ReminderLog,
        task: ReminderDeliveryTask,
        error_message: str,
        now: datetime,
    ) -> None:
        if reminder_log.attempt_count >= settings.reminder_max_attempts:
            await self._repository.mark_reminder_failed(reminder_log, error_message)
            await self._repository.mark_delivery_task_failed(task, error_message)
            await self._send_failure_alert(schedule, reminder_log, error_message)
            return

        next_retry_delay = self._calculate_retry_delay_seconds(reminder_log.attempt_count)
        next_retry_at = now + timedelta(seconds=next_retry_delay)
        await self._repository.mark_reminder_retrying(
            reminder_log,
            error_message=error_message,
            next_retry_at=next_retry_at,
            attempted_at=now,
        )
        await self._repository.requeue_delivery_task(
            task,
            available_at=next_retry_at,
            error_message=error_message,
        )
        await self._queue_backend.enqueue_task(task.id)

    async def _send_failure_alert(
        self,
        schedule: Schedule,
        reminder_log: ReminderLog,
        error_message: str,
    ) -> None:
        dedupe_key = (
            f"reminder_delivery_failed:{schedule.id}:{reminder_log.planned_trigger_at.isoformat()}"
        )
        existing_alert = await self._repository.get_latest_alert_by_dedupe_key(dedupe_key)
        if existing_alert is not None and existing_alert.status in {"pending", "sent"}:
            logger.info(
                "reminder_failure_alert_deduplicated",
                schedule_id=schedule.id,
                reminder_log_id=reminder_log.id,
                dedupe_key=dedupe_key,
            )
            return

        message = (
            f"Reminder delivery failed after {reminder_log.attempt_count} attempts: "
            f"schedule_id={schedule.id}, content={schedule.content}, error={error_message}"
        )
        alert_log = await self._repository.create_alert_log(
            schedule_id=schedule.id,
            reminder_log_id=reminder_log.id,
            alert_type="reminder_delivery_failed",
            alert_channel=settings.reminder_alert_channel,
            message=message,
            dedupe_key=dedupe_key,
        )
        try:
            await self._alert_channel.send(schedule, alert_log)
            await self._repository.mark_alert_sent(alert_log, datetime.now())
        except Exception as exc:  # pragma: no cover
            await self._repository.mark_alert_failed(alert_log, str(exc))
            logger.exception(
                "reminder_failure_alert_failed",
                schedule_id=schedule.id,
                reminder_log_id=reminder_log.id,
            )

    @staticmethod
    def _calculate_retry_delay_seconds(attempt_count: int) -> int:
        base_delay = max(settings.reminder_retry_delay_seconds, 1)
        multiplier = max(settings.reminder_retry_backoff_multiplier, 1.0)
        exponent = max(attempt_count - 1, 0)
        delay = int(base_delay * (multiplier**exponent))
        return min(delay, settings.reminder_retry_max_delay_seconds)
