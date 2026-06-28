import httpx
from structlog import get_logger

from app.channels.base import ReminderChannel
from app.core.config import settings
from app.core.exceptions import ApplicationError
from app.models.schedule import Schedule

logger = get_logger()


class WebhookReminderChannel(ReminderChannel):
    async def send(self, schedule: Schedule) -> None:
        if not settings.reminder_webhook_url:
            raise ApplicationError("REMINDER_WEBHOOK_URL is not configured")

        payload = {
            "schedule_id": schedule.id,
            "content": schedule.content,
            "schedule_time": schedule.schedule_time.isoformat(),
            "message": f"温馨提醒：{schedule.content}的时间到啦，主人！",
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(settings.reminder_webhook_url, json=payload)
            response.raise_for_status()

        logger.info("webhook_reminder_sent", schedule_id=schedule.id)
