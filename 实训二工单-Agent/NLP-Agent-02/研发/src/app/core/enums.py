import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum as _StrEnum
else:  # pragma: no cover
    from enum import Enum

    class _StrEnum(str, Enum):  # noqa: UP042
        pass


class CycleRule(_StrEnum):
    ONCE = "once"
    DAILY = "daily"
    WEEKDAY = "weekday"
    WEEKLY_CUSTOM = "weekly_custom"
    INTERVAL_DAYS = "interval_days"


class ScheduleStatus(_StrEnum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    DONE = "done"


class ReminderStatus(_StrEnum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"
