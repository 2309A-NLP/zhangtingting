from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SchedulerJobRunLogCreate(BaseModel):
    job_id: str = Field(min_length=1, max_length=100)
    job_name: str = Field(min_length=1, max_length=100)
    trigger_name: str | None = Field(default=None, max_length=64)
    started_at: datetime
    finished_at: datetime | None = None
    status: str = Field(min_length=1, max_length=32)
    processed_count: int = Field(default=0, ge=0)
    error_message: str | None = Field(default=None, max_length=500)


class SchedulerJobRunLogRead(SchedulerJobRunLogCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


class SchedulerJobRunLogQuery(BaseModel):
    job_id: str | None = Field(default=None, min_length=1, max_length=100)
    status: str | None = Field(default=None, min_length=1, max_length=32)
    limit: int = Field(default=20, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class SchedulerJobRunLogList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[SchedulerJobRunLogRead]


class SchedulerJobSummary(BaseModel):
    total: int
    running_count: int
    success_count: int
    failed_count: int


class SchedulerJobRuntimeSummaryItem(BaseModel):
    job_id: str
    total_runs: int
    running_count: int
    success_count: int
    failed_count: int
    total_processed_count: int
    avg_processed_count: float
    last_status: str | None = None
    last_started_at: datetime | None = None
    last_finished_at: datetime | None = None
    last_error_message: str | None = None


class SchedulerJobRuntimeSummary(BaseModel):
    total_jobs: int
    items: list[SchedulerJobRuntimeSummaryItem]
