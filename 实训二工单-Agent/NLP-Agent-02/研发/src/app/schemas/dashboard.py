from pydantic import BaseModel


class DashboardScheduleSummary(BaseModel):
    total: int
    active_count: int
    cancelled_count: int
    done_count: int
    due_today_count: int
    overdue_count: int
    upcoming_count: int


class DashboardSessionSummary(BaseModel):
    total: int
    pending_confirmation_count: int
    expired_count: int


class DashboardHistorySummary(BaseModel):
    total: int
    confirm_count: int
    clarify_count: int
    reply_count: int
    llm_source_count: int


class DashboardLLMAuditSummary(BaseModel):
    total: int
    success_count: int
    failed_count: int
    repair_count: int


class DashboardReminderSummary(BaseModel):
    total: int
    pending_count: int
    retrying_count: int
    sent_count: int
    failed_count: int
    max_attempt_reached_count: int
    alert_total: int
    alert_sent_count: int
    alert_failed_count: int


class DashboardWorkerQueueSummary(BaseModel):
    total: int
    queued_count: int
    processing_count: int
    stale_processing_count: int
    done_count: int
    failed_count: int
    redis_enabled: bool = False
    redis_queue_backlog: int = 0


class DashboardSchedulerSummary(BaseModel):
    total: int
    running_count: int
    success_count: int
    failed_count: int


class DashboardOverview(BaseModel):
    schedule: DashboardScheduleSummary
    sessions: DashboardSessionSummary
    history: DashboardHistorySummary
    llm_audit: DashboardLLMAuditSummary
    reminders: DashboardReminderSummary
    worker_queue: DashboardWorkerQueueSummary
    scheduler: DashboardSchedulerSummary
