from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schedule import SchedulerJobLease


class SchedulerJobLeaseRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_job_id(self, job_id: str) -> SchedulerJobLease | None:
        result = await self._session.execute(
            select(SchedulerJobLease).where(SchedulerJobLease.job_id == job_id)
        )
        return result.scalar_one_or_none()

    async def acquire_or_renew(
        self,
        *,
        job_id: str,
        owner_id: str,
        locked_until: datetime,
        now: datetime,
    ) -> bool:
        lease = await self.get_by_job_id(job_id)
        if lease is None:
            lease = SchedulerJobLease(job_id=job_id, owner_id=owner_id, locked_until=locked_until)
            self._session.add(lease)
            await self._session.commit()
            return True

        if lease.owner_id == owner_id or lease.locked_until <= now:
            lease.owner_id = owner_id
            lease.locked_until = locked_until
            await self._session.commit()
            return True

        return False

    async def release(self, *, job_id: str, owner_id: str, now: datetime) -> None:
        lease = await self.get_by_job_id(job_id)
        if lease is None:
            return
        if lease.owner_id != owner_id:
            return
        lease.locked_until = now
        await self._session.commit()
