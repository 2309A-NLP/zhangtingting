from datetime import datetime

from sqlalchemy import Select, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schedule import AgentConversationState
from app.schemas.conversation import ConversationStateCreate


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, payload: ConversationStateCreate) -> AgentConversationState:
        existing = await self.get_by_session_id(payload.session_id)
        if existing is None:
            record = AgentConversationState(**payload.model_dump())
            self._session.add(record)
        else:
            record = existing
            for key, value in payload.model_dump().items():
                setattr(record, key, value)

        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def get_by_session_id(self, session_id: str) -> AgentConversationState | None:
        result = await self._session.execute(
            select(AgentConversationState).where(AgentConversationState.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def list_sessions(
        self,
        *,
        now: datetime,
        include_expired: bool,
        limit: int,
        offset: int,
    ) -> tuple[list[AgentConversationState], int]:
        statement: Select[tuple[AgentConversationState]] = select(AgentConversationState)
        count_statement = select(func.count()).select_from(AgentConversationState)

        if not include_expired:
            active_condition = (
                AgentConversationState.expires_at.is_(None)
                | (AgentConversationState.expires_at > now)
            )
            statement = statement.where(active_condition)
            count_statement = count_statement.where(active_condition)

        statement = (
            statement.order_by(AgentConversationState.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )

        result = await self._session.execute(statement)
        count_result = await self._session.execute(count_statement)
        return list(result.scalars().all()), int(count_result.scalar_one())

    async def export_sessions(
        self,
        *,
        now: datetime,
        include_expired: bool,
    ) -> list[AgentConversationState]:
        statement: Select[tuple[AgentConversationState]] = select(AgentConversationState)

        if not include_expired:
            active_condition = (
                AgentConversationState.expires_at.is_(None)
                | (AgentConversationState.expires_at > now)
            )
            statement = statement.where(active_condition)

        statement = statement.order_by(AgentConversationState.updated_at.desc())
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def clear(self, session_id: str) -> bool:
        record = await self.get_by_session_id(session_id)
        if record is None:
            return False
        await self._session.delete(record)
        await self._session.commit()
        return True

    async def clear_expired(self, now: datetime) -> int:
        result = await self._session.execute(
            select(AgentConversationState).where(
                AgentConversationState.expires_at.is_not(None),
                AgentConversationState.expires_at <= now,
            )
        )
        records = list(result.scalars().all())
        count = 0
        for record in records:
            await self._session.delete(record)
            count += 1
        if count:
            await self._session.commit()
        return count

    async def summarize(self, now: datetime) -> dict[str, int]:
        statement = select(
            func.count().label("total"),
            func.coalesce(
                func.sum(
                    case(
                        (
                            (AgentConversationState.agent_state == "confirm")
                            & AgentConversationState.tool_name.is_not(None)
                            & (
                                AgentConversationState.expires_at.is_(None)
                                | (AgentConversationState.expires_at > now)
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ),
                0,
            ).label("pending_confirmation_count"),
            func.coalesce(
                func.sum(
                    case(
                        (
                            AgentConversationState.expires_at.is_not(None)
                            & (AgentConversationState.expires_at <= now),
                            1,
                        ),
                        else_=0,
                    )
                ),
                0,
            ).label("expired_count"),
        ).select_from(AgentConversationState)

        result = await self._session.execute(statement)
        row = result.one()
        return {
            "total": int(row.total),
            "pending_confirmation_count": int(row.pending_confirmation_count),
            "expired_count": int(row.expired_count),
        }
