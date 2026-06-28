from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.queue import ReminderQueueBackend, get_reminder_queue_backend
from app.repositories.schedule_repository import ScheduleRepository


class MetricsService:
    def __init__(
        self,
        session: AsyncSession,
        queue_backend: ReminderQueueBackend | None = None,
    ) -> None:
        self._repository = ScheduleRepository(session)
        self._queue_backend = queue_backend or get_reminder_queue_backend()

    async def render_prometheus_metrics(self) -> str:
        snapshot = await self._repository.build_metrics_snapshot()
        redis_queue_backlog = await self._queue_backend.get_queue_length()
        '''
        gauge：可增可减的数值（如当前队列长度）
        counter：只增不减的累计值（如总请求数）
        histogram：分布统计（如响应时间）
        summary：分位数统计
        '''
        lines = [
            # 当前活跃的日程总数
            "# HELP schedule_active_total Number of active schedules.",
            "# TYPE schedule_active_total gauge",
            f"schedule_active_total {snapshot['schedule_active_total']}",

            # 提醒日志指标
            # 总日志 → 待处理 → 重试中 → 成功 → 失败 → 超过最大重试
            # 监控重点：
            # pending_total 应该接近 0（说明 Worker 处理及时）
            # failed_total 持续增长 → 需要告警
            # retrying_total 过高 → Worker 处理能力不足
            "# HELP reminder_total Total reminder log records.",
            "# TYPE reminder_total gauge",
            f"reminder_total {snapshot['reminder_total']}",
            "# HELP reminder_pending_total Pending reminder logs.",
            "# TYPE reminder_pending_total gauge",
            f"reminder_pending_total {snapshot['reminder_pending_total']}",
            "# HELP reminder_retrying_total Retrying reminder logs.",
            "# TYPE reminder_retrying_total gauge",
            f"reminder_retrying_total {snapshot['reminder_retrying_total']}",
            "# HELP reminder_sent_total Successfully sent reminder logs.",
            "# TYPE reminder_sent_total gauge",
            f"reminder_sent_total {snapshot['reminder_sent_total']}",
            "# HELP reminder_failed_total Failed reminder logs.",
            "# TYPE reminder_failed_total gauge",
            f"reminder_failed_total {snapshot['reminder_failed_total']}",
            "# HELP reminder_failed_max_attempt_total Failed reminders that reached max attempts.",
            "# TYPE reminder_failed_max_attempt_total gauge",
            f"reminder_failed_max_attempt_total {snapshot['reminder_failed_max_attempt_total']}",

            # 告警日志指标
            # 告警：提醒失败达到阈值后触发的告警通知
            # 监控：alert_total 持续增长说明系统有严重问题
            "# HELP reminder_alert_total Total reminder alert logs.",
            "# TYPE reminder_alert_total gauge",
            f"reminder_alert_total {snapshot['reminder_alert_total']}",
            "# HELP reminder_alert_sent_total Successfully sent reminder alerts.",
            "# TYPE reminder_alert_sent_total gauge",
            f"reminder_alert_sent_total {snapshot['reminder_alert_sent_total']}",
            "# HELP reminder_alert_failed_total Failed reminder alerts.",
            "# TYPE reminder_alert_failed_total gauge",
            f"reminder_alert_failed_total {snapshot['reminder_alert_failed_total']}",

            # 投递任务指标
            # 任务状态分布：总任务 → 排队中 → 处理中 → 僵尸任务 → 已完成 → 失败
            # 关键指标：
            # stale_processing_total（僵尸任务）：Worker 崩溃后留下的锁
            # 正常应该接近 0
            # 持续增长说明 Worker 不稳定
            # queued_total：数据库中的排队任务数
            # processing_total：正在处理的任务数
            "# HELP delivery_task_total Total delivery tasks.",
            "# TYPE delivery_task_total gauge",
            f"delivery_task_total {snapshot['delivery_task_total']}",
            "# HELP delivery_task_queued_total Queued delivery tasks.",
            "# TYPE delivery_task_queued_total gauge",
            f"delivery_task_queued_total {snapshot['delivery_task_queued_total']}",
            "# HELP delivery_task_processing_total Processing delivery tasks.",
            "# TYPE delivery_task_processing_total gauge",
            f"delivery_task_processing_total {snapshot['delivery_task_processing_total']}",
            "# HELP delivery_task_stale_processing_total Stale processing delivery tasks.",
            "# TYPE delivery_task_stale_processing_total gauge",
            f"delivery_task_stale_processing_total {snapshot['delivery_task_stale_processing_total']}",
            "# HELP delivery_task_done_total Completed delivery tasks.",
            "# TYPE delivery_task_done_total gauge",
            f"delivery_task_done_total {snapshot['delivery_task_done_total']}",
            "# HELP delivery_task_failed_total Failed delivery tasks.",
            "# TYPE delivery_task_failed_total gauge",
            f"delivery_task_failed_total {snapshot['delivery_task_failed_total']}",

            # Redis 指标
            # redis_queue_enabled：0 或 1，标识 Redis 是否启用
            # redis_queue_backlog_total：Redis 队列中的积压任务数
            # 监控重点：
            # backlog 持续增长 → Worker 消费能力不足
            # backlog 为 0 但 queued_total 很高 → 入队逻辑有问题
            "# HELP redis_queue_enabled Whether Redis runtime queue is enabled.",
            "# TYPE redis_queue_enabled gauge",
            f"redis_queue_enabled {1 if self._queue_backend.enabled else 0}",
            "# HELP redis_queue_backlog_total Runtime Redis queue backlog.",
            "# TYPE redis_queue_backlog_total gauge",
            f"redis_queue_backlog_total {redis_queue_backlog}",
        ]
        return "\n".join(lines) + "\n"


async def get_metrics_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MetricsService:
    return MetricsService(session, queue_backend=get_reminder_queue_backend())
