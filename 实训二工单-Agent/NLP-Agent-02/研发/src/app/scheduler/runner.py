# APScheduler 的异步调度器，适用于 asyncio 环境
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]

from app.core.config import settings
from app.scheduler.conversation_cleanup_jobs import clear_expired_conversations
from app.scheduler.reminder_jobs import scan_due_reminders
from app.scheduler.worker_jobs import process_reminder_delivery_tasks

'''
任务                            	   频率	            职责
scan_due_reminders	               每 1 分钟	      扫描有没有到期的日程，创建提醒任务入队
clear_expired_conversations	       每 10 分钟	  清理过期的会话状态
process_reminder_delivery_tasks	   每 10 秒	      从任务队列取任务并发送提醒
'''

def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=settings.timezone)
    scheduler.add_job(scan_due_reminders, "interval", minutes=1, id="scan_due_reminders")
    scheduler.add_job(
        clear_expired_conversations, "interval", minutes=10, id="clear_expired_conversations"
    )
    return scheduler


def create_worker_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=settings.timezone)
    scheduler.add_job(
        process_reminder_delivery_tasks,
        "interval",
        seconds=settings.worker_poll_interval_seconds,
        id="process_reminder_delivery_tasks",
    )
    return scheduler
