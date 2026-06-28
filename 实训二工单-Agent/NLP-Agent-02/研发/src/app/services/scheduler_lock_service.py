from datetime import datetime, timedelta
from typing import Annotated
# 调度器分布式锁服务，用于在分布式环境中确保同一个定时任务在同一时间只有一个实例在运行。
# 它通过数据库实现分布式锁，防止多个服务实例同时执行同一个调度任务。
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db_session
from app.repositories.scheduler_lease_repository import SchedulerJobLeaseRepository
from app.schemas.scheduler_runtime import SchedulerLeaseRead

'''
SchedulerLockService
├── 初始化（依赖注入）
│   └── SchedulerJobLeaseRepository（数据访问层）
├── 工厂方法
│   └── from_session() → 从数据库会话创建 Service
├── 锁操作
│   ├── acquire() → 获取/续期锁
│   ├── release() → 释放锁
│   └── get_lease() → 查询锁状态
└── 配置依赖
    ├── settings.scheduler_lock_enabled（是否启用锁）
    ├── settings.scheduler_lock_ttl_seconds（锁有效期）
    └── settings.scheduler_lock_owner（锁持有者标识）
'''

class SchedulerLockService:
    def __init__(self, repository: SchedulerJobLeaseRepository) -> None:
        self._repository = repository

    @classmethod
    def from_session(cls, session: AsyncSession) -> "SchedulerLockService":
        return cls(SchedulerJobLeaseRepository(session))

    # 获取/续期锁
    # 核心操作：
    # 如果锁不存在 → 创建锁（获取成功）
    # 如果锁存在且属于当前实例 → 续期（延长有效期）
    # 如果锁存在且属于其他实例 → 检查是否过期
    # 已过期 → 接管锁（获取成功）
    # 未过期 → 获取失败
    async def acquire(self, job_id: str, *, now: datetime | None = None) -> bool:
        if not settings.scheduler_lock_enabled:
            return True
        reference_time = now or datetime.now()
        locked_until = reference_time + timedelta(seconds=settings.scheduler_lock_ttl_seconds)
        return await self._repository.acquire_or_renew(
            job_id=job_id,
            owner_id=settings.scheduler_lock_owner,
            locked_until=locked_until,
            now=reference_time,
        )

    # 释放锁
    async def release(self, job_id: str, *, now: datetime | None = None) -> None:
        if not settings.scheduler_lock_enabled:
            return
        reference_time = now or datetime.now()
        await self._repository.release(
            job_id=job_id,
            owner_id=settings.scheduler_lock_owner,
            now=reference_time,
        )

    # 查询锁状态
    async def get_lease(self, job_id: str) -> SchedulerLeaseRead | None:
        lease = await self._repository.get_by_job_id(job_id)
        if lease is None:
            return None
        return SchedulerLeaseRead.model_validate(lease)


async def get_scheduler_lock_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> SchedulerLockService:
    return SchedulerLockService.from_session(session)
