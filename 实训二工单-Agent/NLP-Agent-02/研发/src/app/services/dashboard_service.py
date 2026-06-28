from datetime import datetime
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db_session
from app.queue import ReminderQueueBackend, get_reminder_queue_backend
from app.repositories.conversation_history_repository import ConversationHistoryRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.llm_audit_repository import LLMAuditLogRepository
from app.repositories.schedule_repository import ScheduleRepository
from app.repositories.scheduler_audit_repository import SchedulerJobRunLogRepository
from app.schemas.dashboard import (
    DashboardHistorySummary,
    DashboardLLMAuditSummary,
    DashboardOverview,
    DashboardReminderSummary,
    DashboardSchedulerSummary,
    DashboardScheduleSummary,
    DashboardSessionSummary,
    DashboardWorkerQueueSummary,
)
# ScheduleQuery：日程查询条件（空查询表示全部）
from app.schemas.schedule import ScheduleQuery

'''
DashboardService
├── 初始化（依赖注入）
│   ├── ScheduleRepository（日程数据访问层）
│   ├── ConversationRepository（会话确认数据访问层）
│   ├── ConversationHistoryRepository（对话历史数据访问层）
│   ├── LLMAuditLogRepository（LLM审计日志数据访问层）
│   ├── SchedulerJobRunLogRepository（调度器任务运行日志数据访问层）
│   └── ReminderQueueBackend（消息队列后端）
├── 核心方法
│   └── get_overview() → 获取仪表盘总览（聚合7个数据源）
├── 数据聚合（7个维度）
│   ├── 日程统计（ScheduleRepository.summarize）
│   │   ├── total（总数）
│   │   ├── active（活跃）
│   │   ├── paused（暂停）
│   │   ├── archived（归档）
│   │   ├── due_now（当前到期）
│   │   ├── upcoming（即将到来）
│   │   └── overdue（已过期）
│   ├── 会话统计（ConversationRepository.summarize）
│   │   ├── total（总会话）
│   │   ├── pending_confirmation（待确认）
│   │   └── expired（已过期）
│   ├── 历史统计（ConversationHistoryRepository.summarize）
│   │   ├── total（总记录数）
│   │   └── by_intent（按意图分组）
│   ├── LLM审计统计（LLMAuditLogRepository.summarize）
│   │   ├── total（总调用）
│   │   ├── success（成功）
│   │   └── failed（失败）
│   ├── 提醒统计（ScheduleRepository.summarize_reminder_logs）
│   │   ├── total（总提醒）
│   │   ├── pending_count（待处理）
│   │   ├── retrying_count（重试中）
│   │   ├── sent_count（已发送）
│   │   ├── failed_count（失败）
│   │   └── max_attempt_reached_count（达到最大重试）
│   ├── 告警统计（ScheduleRepository.summarize_alert_logs）
│   │   ├── alert_total（总告警）
│   │   ├── alert_sent_count（已发送）
│   │   └── alert_failed_count（失败）
│   ├── 队列统计（ScheduleRepository.summarize_delivery_tasks + QueueBackend）
│   │   ├── total（总任务）
│   │   ├── pending（待处理）
│   │   ├── processing（处理中）
│   │   ├── success（成功）
│   │   ├── failed（失败）
│   │   ├── locked_by_worker（被Worker锁定）
│   │   ├── stale_locks（过期锁）
│   │   ├── redis_enabled（Redis是否启用）
│   │   └── redis_queue_backlog（Redis队列积压数）
│   └── 调度器统计（SchedulerJobRunLogRepository.summarize）
│       ├── total_runs（总运行次数）
│       ├── success（成功）
│       └── failed（失败）
├── 配置依赖
│   └── settings.worker_lock_timeout_seconds（Worker锁超时配置）
└── FastAPI 依赖注入
    └── get_dashboard_service() → 从会话创建 Service 实例
'''

class DashboardService:
    def __init__(
        self,
        session: AsyncSession,
        queue_backend: ReminderQueueBackend | None = None,
    ) -> None:
        # Dashboard 服务需要访问多个表，所以持有多个 Repository
        self._schedule_repository = ScheduleRepository(session)
        self._conversation_repository = ConversationRepository(session)
        self._conversation_history_repository = ConversationHistoryRepository(session)
        self._llm_audit_repository = LLMAuditLogRepository(session)
        self._scheduler_audit_repository = SchedulerJobRunLogRepository(session)
        self._queue_backend = queue_backend or get_reminder_queue_backend()

    # 获取总览
    async def get_overview(self) -> DashboardOverview:
        now = datetime.now()
        # 日程统计
        schedule_summary_raw = await self._schedule_repository.summarize(ScheduleQuery(), now=now)
        # 会话统计
        session_summary_raw = await self._conversation_repository.summarize(now=now)
        # 会话历史统计
        history_summary_raw = await self._conversation_history_repository.summarize()
        #  LLM审计统计
        llm_audit_summary_raw = await self._llm_audit_repository.summarize()
        # 提醒统计
        reminder_summary_raw = await self._schedule_repository.summarize_reminder_logs()
        # 告警统计
        alert_summary_raw = await self._schedule_repository.summarize_alert_logs()
        # 队列统计
        delivery_queue_summary_raw = await self._schedule_repository.summarize_delivery_tasks(
            now=now,
            worker_lock_timeout_seconds=settings.worker_lock_timeout_seconds,
        )
        # 队列是否启用
        delivery_queue_summary_raw["redis_enabled"] = self._queue_backend.enabled
        # Redis 中积压的任务数
        delivery_queue_summary_raw["redis_queue_backlog"] = await self._queue_backend.get_queue_length()
        # 调度器统计
        scheduler_summary_raw = await self._scheduler_audit_repository.summarize()

        return DashboardOverview(
            schedule=DashboardScheduleSummary(**schedule_summary_raw),
            sessions=DashboardSessionSummary(**session_summary_raw),
            history=DashboardHistorySummary(**history_summary_raw),
            llm_audit=DashboardLLMAuditSummary(**llm_audit_summary_raw),
            reminders=DashboardReminderSummary(
                total=reminder_summary_raw["total"],
                pending_count=reminder_summary_raw["pending_count"],
                retrying_count=reminder_summary_raw["retrying_count"],
                sent_count=reminder_summary_raw["sent_count"],
                failed_count=reminder_summary_raw["failed_count"],
                max_attempt_reached_count=reminder_summary_raw["max_attempt_reached_count"],
                alert_total=alert_summary_raw["total"],
                alert_sent_count=alert_summary_raw["sent_count"],
                alert_failed_count=alert_summary_raw["failed_count"],
            ),
            worker_queue=DashboardWorkerQueueSummary(**delivery_queue_summary_raw),
            scheduler=DashboardSchedulerSummary(**scheduler_summary_raw),
        )


async def get_dashboard_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> DashboardService:
    return DashboardService(session, queue_backend=get_reminder_queue_backend())
