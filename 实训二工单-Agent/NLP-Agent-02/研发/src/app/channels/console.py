from structlog import get_logger

from app.channels.base import ReminderChannel
from app.models.schedule import Schedule

logger = get_logger()


class ConsoleReminderChannel(ReminderChannel):
    async def send(self, schedule: Schedule) -> None:
        logger.info(
            "console_reminder_sent",
            schedule_id=schedule.id,
            content=schedule.content,
            reminder_message=f"温馨提醒：{schedule.content}的时间到啦，主人！",
        )
