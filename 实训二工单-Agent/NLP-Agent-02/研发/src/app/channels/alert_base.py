from abc import ABC, abstractmethod

from app.models.schedule import ReminderAlertLog, Schedule


class ReminderAlertChannel(ABC):
    @abstractmethod
    async def send(self, schedule: Schedule, alert_log: ReminderAlertLog) -> None:
        raise NotImplementedError
