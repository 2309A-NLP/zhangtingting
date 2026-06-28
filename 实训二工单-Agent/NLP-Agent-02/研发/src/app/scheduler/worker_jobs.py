from datetime import datetime
# 从投递任务队列中取出 queued 状态的任务，认领并执行发送逻辑
from structlog import get_logger

from app.core.database import async_session_factory
from app.services.reminder_worker_service import ReminderWorkerService
from app.services.scheduler_audit_service import SchedulerJobRunLogService
from app.services.scheduler_lock_service import SchedulerLockService

logger = get_logger()


async def process_reminder_delivery_tasks() -> None:
    started_at = datetime.now()
    logger.info("process_reminder_delivery_tasks_started")
    async with async_session_factory() as session:
        lock_service = SchedulerLockService.from_session(session)
        lock_acquired = await lock_service.acquire("process_reminder_delivery_tasks", now=started_at)
        if not lock_acquired:
            logger.info("process_reminder_delivery_tasks_skipped_by_lease_lock")
            return
        audit_service = SchedulerJobRunLogService.from_session(session)
        run_log = await audit_service.start_run(
            job_id="process_reminder_delivery_tasks",
            job_name="process_reminder_delivery_tasks",
            trigger_name="interval",
            started_at=started_at,
        )
        service = ReminderWorkerService(session)
        try:
            processed_count = await service.process_delivery_tasks()
            await audit_service.finish_run(
                run_log,
                finished_at=datetime.now(),
                status="success",
                processed_count=processed_count,
                error_message=None,
            )
            logger.info("process_reminder_delivery_tasks_completed", processed_count=processed_count)
        except Exception as exc:  # pragma: no cover
            await audit_service.finish_run(
                run_log,
                finished_at=datetime.now(),
                status="failed",
                processed_count=0,
                error_message=str(exc),
            )
            logger.exception("process_reminder_delivery_tasks_failed")
            raise
        finally:
            await lock_service.release("process_reminder_delivery_tasks", now=datetime.now())
