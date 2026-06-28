from structlog import get_logger

from app.channels.alert_base import ReminderAlertChannel
from app.models.schedule import ReminderAlertLog, Schedule

logger = get_logger()


class ConsoleReminderAlertChannel(ReminderAlertChannel):
    async def send(self, schedule: Schedule, alert_log: ReminderAlertLog) -> None:
        logger.warning(
            "console_reminder_alert_sent",
            schedule_id=schedule.id,
            reminder_log_id=alert_log.reminder_log_id,
            alert_type=alert_log.alert_type,
            message=alert_log.message,
        )
