from __future__ import annotations

from datetime import date as date_type
from datetime import datetime as datetime_type
from datetime import time as time_type
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.enums import CycleRule, ReminderStatus, ScheduleStatus


class ScheduleBase(BaseModel):
    content: str = Field(min_length=1, max_length=255)
    schedule_date: date_type | None = None
    schedule_time: time_type
    cycle_rule: CycleRule = Field(default=CycleRule.ONCE)
    cycle_value: str | None = None
    source_text: str | None = None

    @model_validator(mode="after")
    def validate_cycle_rule(self) -> ScheduleBase:
        if self.cycle_rule == CycleRule.WEEKLY_CUSTOM and not self.cycle_value:
            raise ValueError("cycle_value is required when cycle_rule is weekly_custom")
        if self.cycle_rule == CycleRule.INTERVAL_DAYS and not self.cycle_value:
            raise ValueError("cycle_value is required when cycle_rule is interval_days")
        if self.cycle_rule == CycleRule.ONCE and self.schedule_date is None:
            raise ValueError("schedule_date is required when cycle_rule is once")
        return self


class ScheduleCreate(ScheduleBase):
    pass


class ScheduleUpdate(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=255)
    schedule_date: date_type | None = None
    schedule_time: time_type | None = None
    cycle_rule: CycleRule | None = None
    cycle_value: str | None = None
    source_text: str | None = None
    status: ScheduleStatus | None = None


class ScheduleQuery(BaseModel):
    date: date_type | None = None
    start_date: date_type | None = None
    end_date: date_type | None = None
    status: ScheduleStatus | None = None
    statuses: list[ScheduleStatus] | None = None
    schedule_time_start: time_type | None = None
    schedule_time_end: time_type | None = None
    keyword: str | None = Field(default=None, min_length=1, max_length=255)
    sort_by: Literal[
        "id",
        "created_at",
        "updated_at",
        "schedule_date",
        "schedule_time",
        "next_trigger_at",
    ] = "created_at"
    sort_order: Literal["asc", "desc"] = "desc"
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_query_ranges(self) -> ScheduleQuery:
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date cannot be later than end_date")
        if self.schedule_time_start and self.schedule_time_end and self.schedule_time_start > self.schedule_time_end:
            raise ValueError("schedule_time_start cannot be later than schedule_time_end")
        return self


class ScheduleRead(ScheduleBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: ScheduleStatus
    next_trigger_at: datetime_type | None = None
    created_at: datetime_type
    updated_at: datetime_type


class ScheduleList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ScheduleRead]


class ScheduleSummary(BaseModel):
    total: int
    active_count: int
    cancelled_count: int
    done_count: int
    due_today_count: int
    overdue_count: int
    upcoming_count: int


class ReminderLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    schedule_id: int
    planned_trigger_at: datetime_type
    reminded_at: datetime_type | None = None
    status: ReminderStatus
    attempt_count: int
    last_attempt_at: datetime_type | None = None
    next_retry_at: datetime_type | None = None
    error_message: str | None = None
    created_at: datetime_type
    updated_at: datetime_type


class ReminderLogQuery(BaseModel):
    schedule_id: int | None = None
    status: ReminderStatus | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ReminderLogList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ReminderLogRead]


class ReminderAlertLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    schedule_id: int
    reminder_log_id: int | None = None
    alert_type: str
    alert_channel: str
    status: str
    message: str
    dedupe_key: str | None = None
    sent_at: datetime_type | None = None
    error_message: str | None = None
    created_at: datetime_type
    updated_at: datetime_type


class ReminderAlertLogQuery(BaseModel):
    schedule_id: int | None = None
    status: str | None = None
    alert_type: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ReminderAlertLogList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ReminderAlertLogRead]


class ReminderReliabilitySummary(BaseModel):
    total_logs: int
    pending_count: int
    retrying_count: int
    sent_count: int
    failed_count: int
    max_attempt_reached_count: int
    alert_total: int
    alert_sent_count: int
    alert_failed_count: int


class ReminderDeliveryTaskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    schedule_id: int
    reminder_log_id: int
    task_type: str
    status: str
    available_at: datetime_type
    locked_by: str | None = None
    locked_at: datetime_type | None = None
    last_error_message: str | None = None
    created_at: datetime_type
    updated_at: datetime_type


class ReminderDeliveryTaskQuery(BaseModel):
    schedule_id: int | None = None
    status: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ReminderDeliveryTaskList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ReminderDeliveryTaskRead]


class ReminderDeliveryQueueSummary(BaseModel):
    total: int
    queued_count: int
    processing_count: int
    stale_processing_count: int
    done_count: int
    failed_count: int
    redis_enabled: bool = False
    redis_queue_backlog: int = 0
