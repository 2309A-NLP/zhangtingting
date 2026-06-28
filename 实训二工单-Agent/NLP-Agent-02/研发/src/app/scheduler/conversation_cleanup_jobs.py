from datetime import datetime

from structlog import get_logger

from app.core.database import async_session_factory
from app.services.conversation_service import ConversationService
from app.services.scheduler_audit_service import SchedulerJobRunLogService
from app.services.scheduler_lock_service import SchedulerLockService

logger = get_logger()


async def clear_expired_conversations() -> None:
    started_at = datetime.now()
    logger.info("clear_expired_conversations_started")
    async with async_session_factory() as session:
        lock_service = SchedulerLockService.from_session(session)
        lock_acquired = await lock_service.acquire("clear_expired_conversations", now=started_at)
        if not lock_acquired:
            logger.info("clear_expired_conversations_skipped_by_lease_lock")
            return
        audit_service = SchedulerJobRunLogService.from_session(session)
        run_log = await audit_service.start_run(
            job_id="clear_expired_conversations",
            job_name="clear_expired_conversations",
            trigger_name="interval",
            started_at=started_at,
        )
        service = ConversationService.from_session(session)
        try:
            cleaned_count = await service.clear_expired_confirmations()
            await audit_service.finish_run(
                run_log,
                finished_at=datetime.now(),
                status="success",
                processed_count=cleaned_count,
                error_message=None,
            )
            logger.info("clear_expired_conversations_completed", cleaned_count=cleaned_count)
        except Exception as exc:  # pragma: no cover
            await audit_service.finish_run(
                run_log,
                finished_at=datetime.now(),
                status="failed",
                processed_count=0,
                error_message=str(exc),
            )
            logger.exception("clear_expired_conversations_failed")
            raise
        finally:
            await lock_service.release("clear_expired_conversations", now=datetime.now())
