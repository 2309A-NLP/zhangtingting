from datetime import date, datetime, time
# 日程提醒系统数据模型
from sqlalchemy import Date, DateTime, ForeignKey, String, Text, Time, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import CycleRule, ReminderStatus, ScheduleStatus
from app.models.base import Base, TimestampMixin

'''
┌─────────────────────────────────────────────────────────────────────────────┐
│                              日程提醒系统                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                          │
│  │   Schedule   │ ◄─── 核心：日程表                                        │
│  │   (日程)     │                                                          │
│  └──────┬───────┘                                                          │
│         │ 1:N                                                              │
│         ▼                                                                  │
│  ┌──────────────┐                                                          │
│  │ ReminderLog  │ ◄─── 提醒日志（每次触发）                                │
│  │  (提醒日志)  │                                                          │
│  └──────┬───────┘                                                          │
│         │ 1:N                                                              │
│         ├──────────────────────┬──────────────────────┐                    │
│         ▼                      ▼                      ▼                    │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐            │
│  │ReminderDelivery  │  │ ReminderAlert │  │AgentConversation │            │
│  │     Task         │  │     Log       │  │    History       │            │
│  │   (投递任务)     │  │   (告警日志)  │  │  (对话历史)      │            │
│  └──────────────────┘  └──────────────┘  └──────────────────┘            │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐            │
│  │  LLMAuditLog     │  │SchedulerJob  │  │AdminAccessAudit  │            │
│  │  (LLM审计日志)   │  │   RunLog     │  │      Log         │            │
│  │                  │  │ (调度器运行)  │  │  (管理员审计)    │            │
│  └──────────────────┘  └──────────────┘  └──────────────────┘            │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────┐                                    │
│  │AgentConversation │  │SchedulerJob  │                                    │
│  │     State        │  │    Lease     │                                    │
│  │  (会话状态)      │  │  (任务租约)  │                                    │
│  └──────────────────┘  └──────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘

1️⃣ Schedule — 日程表（核心）
字段	类型	说明
id	int	主键
content	str	日程内容（如 "开会"）
schedule_date	date | None	日程日期
schedule_time	time	日程时间（必填）
cycle_rule	str	循环规则（once/interval_days 等）
cycle_value	str | None	循环值（如 "1" 表示每隔 1 天）
status	str	状态（active/done/cancelled）
source_text	str | None	用户原始输入
next_trigger_at	datetime | None	下次触发时间
关系： reminder_logs (1:N)

2️⃣ ReminderLog — 提醒日志
字段	类型	说明
id	int	主键
schedule_id	int	FK → schedule.id
planned_trigger_at	datetime	计划触发时间
reminded_at	datetime | None	实际提醒时间
status	str	pending/sent/failed/retrying
attempt_count	int	尝试次数（默认 1）
last_attempt_at	datetime | None	最后尝试时间
next_retry_at	datetime | None	下次重试时间
error_message	str | None	错误信息
约束： (schedule_id, planned_trigger_at) 唯一

关系： schedule (N:1)

3️⃣ ReminderDeliveryTask — 投递任务
字段	类型	说明
id	int	主键
schedule_id	int	FK → schedule.id
reminder_log_id	int	FK → reminder_log.id
task_type	str	任务类型（send/retry/cancel）
status	str	queued/processing/done/failed
available_at	datetime	可执行时间
locked_by	str | None	锁定 Worker ID
locked_at	datetime | None	锁定时间
last_error_message	str | None	错误信息
4️⃣ ReminderAlertLog — 告警日志
字段	类型	说明
id	int	主键
schedule_id	int	FK → schedule.id
reminder_log_id	int | None	FK → reminder_log.id
alert_type	str	告警类型（reminder/warning/error）
alert_channel	str	渠道（email/sms/push）
status	str	pending/sent/failed
message	str	告警内容
dedupe_key	str | None	去重键（有索引）
sent_at	datetime | None	发送时间
error_message	str | None	错误信息
5️⃣ AgentConversationState — 会话状态
字段	类型	说明
id	int	主键
session_id	str	会话 ID（唯一）
intent	str	意图
agent_state	str	Agent 状态
tool_name	str | None	工具名
tool_arguments_json	str | None	工具参数（JSON）
user_message	str | None	用户消息
expires_at	datetime | None	过期时间
6️⃣ AgentConversationHistory — 对话历史
字段	类型	说明
id	int	主键
session_id	str	会话 ID（有索引）
user_input	str	用户输入
confirmed	bool	是否已确认
intent	str	意图
agent_state	str	Agent 状态
tool_name	str | None	工具名
tool_arguments_json	str | None	工具参数
missing_fields_json	str | None	缺失字段
execution_result_json	str | None	执行结果
user_message	str | None	系统回复
7️⃣ LLMAuditLog — LLM 审计日志
字段	类型	说明
id	int	主键
session_id	str	会话 ID
user_input	str	用户输入
parser_stage	str	解析阶段（plan/repair）
provider	str | None	LLM 提供商
model_name	str | None	模型名称
success	bool	是否成功
request_payload_json	str | None	请求内容
raw_response_text	str | None	原始响应
parsed_response_json	str | None	解析后的响应
error_message	str | None	错误信息
8️⃣ SchedulerJobRunLog — 调度器运行日志
字段	类型	说明
id	int	主键
job_id	str	任务 ID
job_name	str	任务名称
started_at	datetime	开始时间
finished_at	datetime | None	结束时间
status	str	running/success/failed
processed_count	int	处理数量
error_message	str | None	错误信息
9️⃣ SchedulerJobLease — 调度器任务租约
字段	类型	说明
id	int	主键
job_id	str	任务 ID（唯一）
owner_id	str	所有者 ID
locked_until	datetime	锁定到期时间
🔟 AdminAccessAuditLog — 管理员审计日志
字段	类型	说明
id	int	主键
path	str	请求路径
method	str	HTTP 方法
client_host	str | None	客户端 IP
request_id	str | None	请求 ID
access_granted	bool	是否授权
auth_mode	str	认证方式
failure_reason	str | None	失败原因
'''

class Schedule(TimestampMixin, Base):
    __tablename__ = "schedule"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(String(255), nullable=False)
    schedule_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    schedule_time: Mapped[time] = mapped_column(Time, nullable=False)
    cycle_rule: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text(f"'{CycleRule.ONCE.value}'")
    )
    cycle_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text(f"'{ScheduleStatus.ACTIVE.value}'")
    )
    source_text: Mapped[str | None] = mapped_column(String(500), nullable=True)
    next_trigger_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    reminder_logs: Mapped[list["ReminderLog"]] = relationship(
        back_populates="schedule", cascade="all, delete-orphan"
    )


class ReminderLog(TimestampMixin, Base):
    __tablename__ = "reminder_log"
    __table_args__ = (
        UniqueConstraint("schedule_id", "planned_trigger_at", name="uq_reminder_log_schedule_trigger"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    schedule_id: Mapped[int] = mapped_column(ForeignKey("schedule.id"), nullable=False, index=True)
    planned_trigger_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    reminded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text(f"'{ReminderStatus.PENDING.value}'")
    )
    attempt_count: Mapped[int] = mapped_column(nullable=False, server_default=text("1"))
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    schedule: Mapped[Schedule] = relationship(back_populates="reminder_logs")


class AgentConversationState(TimestampMixin, Base):
    __tablename__ = "agent_conversation_state"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    intent: Mapped[str] = mapped_column(String(32), nullable=False)
    agent_state: Mapped[str] = mapped_column(String(32), nullable=False)
    tool_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_arguments_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)


class AgentConversationHistory(TimestampMixin, Base):
    __tablename__ = "agent_conversation_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    confirmed: Mapped[bool] = mapped_column(nullable=False, server_default=text("0"))
    context_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    parser_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    intent: Mapped[str] = mapped_column(String(32), nullable=False)
    agent_state: Mapped[str] = mapped_column(String(32), nullable=False)
    tool_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_arguments_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    missing_fields_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggested_inputs_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_id: Mapped[int | None] = mapped_column(nullable=True)
    execution_result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class LLMAuditLog(TimestampMixin, Base):
    __tablename__ = "llm_audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    parser_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    success: Mapped[bool] = mapped_column(nullable=False, server_default=text("0"))
    request_payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_response_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)


class AdminAccessAuditLog(TimestampMixin, Base):
    __tablename__ = "admin_access_audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(16), nullable=False)
    client_host: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    access_granted: Mapped[bool] = mapped_column(nullable=False, server_default=text("0"))
    auth_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)


class SchedulerJobRunLog(TimestampMixin, Base):
    __tablename__ = "scheduler_job_run_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    job_name: Mapped[str] = mapped_column(String(100), nullable=False)
    trigger_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'running'"))
    processed_count: Mapped[int] = mapped_column(nullable=False, server_default=text("0"))
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)


class ReminderAlertLog(TimestampMixin, Base):
    __tablename__ = "reminder_alert_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    schedule_id: Mapped[int] = mapped_column(ForeignKey("schedule.id"), nullable=False, index=True)
    reminder_log_id: Mapped[int | None] = mapped_column(ForeignKey("reminder_log.id"), nullable=True, index=True)
    alert_type: Mapped[str] = mapped_column(String(64), nullable=False)
    alert_channel: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'pending'"))
    message: Mapped[str] = mapped_column(String(500), nullable=False)
    dedupe_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)


class SchedulerJobLease(TimestampMixin, Base):
    __tablename__ = "scheduler_job_lease"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    owner_id: Mapped[str] = mapped_column(String(100), nullable=False)
    locked_until: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)


class ReminderDeliveryTask(TimestampMixin, Base):
    __tablename__ = "reminder_delivery_task"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    schedule_id: Mapped[int] = mapped_column(ForeignKey("schedule.id"), nullable=False, index=True)
    reminder_log_id: Mapped[int] = mapped_column(ForeignKey("reminder_log.id"), nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'send'"))
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'queued'"))
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    locked_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    last_error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
