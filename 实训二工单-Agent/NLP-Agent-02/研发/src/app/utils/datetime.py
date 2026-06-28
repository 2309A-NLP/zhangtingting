from datetime import date, datetime, time, timedelta

from app.core.enums import CycleRule


def combine_schedule_datetime(schedule_date: date | None, schedule_time: time) -> datetime | None:
    if schedule_date is None:
        return None
    return datetime.combine(schedule_date, schedule_time)


def calculate_next_trigger_at(
    *,
    schedule_date: date | None,
    schedule_time: time,
    cycle_rule: CycleRule,
    cycle_value: str | None = None,
    now: datetime | None = None,
) -> datetime | None:
    current_time = now or datetime.now()

    if cycle_rule == CycleRule.ONCE:
        return combine_schedule_datetime(schedule_date, schedule_time)

    if cycle_rule == CycleRule.DAILY:
        base_date = schedule_date or current_time.date()
        candidate = datetime.combine(base_date, schedule_time)
        if candidate <= current_time:
            candidate = datetime.combine(candidate.date() + timedelta(days=1), schedule_time)
        return candidate

    if cycle_rule == CycleRule.WEEKDAY:
        candidate_date = schedule_date or current_time.date()
        for day_offset in range(0, 8):
            checking_date = candidate_date + timedelta(days=day_offset)
            if checking_date.weekday() < 5:
                candidate = datetime.combine(checking_date, schedule_time)
                if candidate > current_time:
                    return candidate
        return None

    if cycle_rule == CycleRule.WEEKLY_CUSTOM:
        if not cycle_value:
            return None
        valid_weekdays = parse_cycle_value(cycle_value)
        candidate_date = schedule_date or current_time.date()
        for day_offset in range(0, 15):
            checking_date = candidate_date + timedelta(days=day_offset)
            weekday_number = checking_date.isoweekday()
            if weekday_number in valid_weekdays:
                candidate = datetime.combine(checking_date, schedule_time)
                if candidate > current_time:
                    return candidate
        return None

    if cycle_rule == CycleRule.INTERVAL_DAYS:
        if not cycle_value:
            return None
        try:
            interval_days = int(cycle_value)
        except ValueError:
            return None
        if interval_days <= 0:
            return None
        base_date = schedule_date or current_time.date()
        candidate = datetime.combine(base_date, schedule_time)
        if candidate <= current_time:
            candidate = candidate + timedelta(days=interval_days)
        return candidate

    return None


def calculate_following_trigger_at(
    *,
    current_trigger_at: datetime | None,
    schedule_time: time,
    cycle_rule: CycleRule,
    cycle_value: str | None = None,
) -> datetime | None:
    if current_trigger_at is None:
        return None
    if cycle_rule == CycleRule.ONCE:
        return None
    return calculate_next_trigger_at(
        schedule_date=None,
        schedule_time=schedule_time,
        cycle_rule=cycle_rule,
        cycle_value=cycle_value,
        now=current_trigger_at,
    )


def parse_cycle_value(cycle_value: str) -> set[int]:
    values = {int(item.strip()) for item in cycle_value.split(",") if item.strip()}
    return {item for item in values if 1 <= item <= 7}
