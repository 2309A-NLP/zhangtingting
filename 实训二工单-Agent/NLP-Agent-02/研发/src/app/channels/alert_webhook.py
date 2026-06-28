import httpx

from app.channels.alert_base import ReminderAlertChannel
from app.core.config import settings
from app.core.exceptions import ApplicationError
from app.models.schedule import ReminderAlertLog, Schedule


class WebhookReminderAlertChannel(ReminderAlertChannel):
    async def send(self, schedule: Schedule, alert_log: ReminderAlertLog) -> None:
        if not settings.reminder_alert_webhook_url:
            raise ApplicationError("REMINDER_ALERT_WEBHOOK_URL is not configured")

        payload = {
            "schedule_id": schedule.id,
            "reminder_log_id": alert_log.reminder_log_id,
            "alert_type": alert_log.alert_type,
            "message": alert_log.message,
            "status": alert_log.status,
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(settings.reminder_alert_webhook_url, json=payload)
            response.raise_for_status()
