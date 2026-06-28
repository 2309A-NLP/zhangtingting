from abc import ABC, abstractmethod

from app.models.schedule import Schedule


class ReminderChannel(ABC):
    @abstractmethod
    async def send(self, schedule: Schedule) -> None:
        raise NotImplementedError
