from datetime import datetime
# 作为调度器的一个定时任务，扫描所有已到触发时间的活跃日程，
# 为每个到期日程创建提醒日志和投递任务，交给 Worker 异步处理。
from structlog import get_logger

from app.core.database import async_session_factory
from app.services.reminder_service import ReminderService
from app.services.scheduler_audit_service import SchedulerJobRunLogService
from app.services.scheduler_lock_service import SchedulerLockService

logger = get_logger()


async def scan_due_reminders() -> None:
    started_at = datetime.now()
    logger.info("scan_due_reminders_started")
    # 获取数据库会话
    async with async_session_factory() as session:
        lock_service = SchedulerLockService.from_session(session)
        lock_acquired = await lock_service.acquire("scan_due_reminders", now=started_at)
        if not lock_acquired:
            logger.info("scan_due_reminders_skipped_by_lease_lock")
            return
        # 创建审计日志
        audit_service = SchedulerJobRunLogService.from_session(session)
        run_log = await audit_service.start_run(
            job_id="scan_due_reminders",
            job_name="scan_due_reminders",
            trigger_name="interval",
            started_at=started_at,
        )
        # 执行核心业务逻辑
        service = ReminderService(session)
        try:
            queued_count = await service.scan_and_enqueue_due_reminders()
            await audit_service.finish_run(
                run_log,
                finished_at=datetime.now(),
                status="success",
                processed_count=queued_count,
                error_message=None,
            )
            logger.info("scan_due_reminders_completed", queued_count=queued_count)
        except Exception as exc:  # pragma: no cover
            await audit_service.finish_run(
                run_log,
                finished_at=datetime.now(),
                status="failed",
                processed_count=0,
                error_message=str(exc),
            )
            logger.exception("scan_due_reminders_failed")
            raise
        finally:
            await lock_service.release("scan_due_reminders", now=datetime.now())
