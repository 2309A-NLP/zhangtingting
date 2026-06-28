from sqlalchemy import Select, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schedule import AgentConversationHistory
from app.schemas.conversation_history import ConversationHistoryCreate, ConversationHistoryQuery


class ConversationHistoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, payload: ConversationHistoryCreate) -> AgentConversationHistory:
        record = AgentConversationHistory(**payload.model_dump())
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def list_by_session_id(self, session_id: str) -> list[AgentConversationHistory]:
        statement: Select[tuple[AgentConversationHistory]] = (
            select(AgentConversationHistory)
            .where(AgentConversationHistory.session_id == session_id)
            .order_by(AgentConversationHistory.id.asc())
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def list_logs(
        self,
        query: ConversationHistoryQuery,
    ) -> tuple[list[AgentConversationHistory], int]:
        statement: Select[tuple[AgentConversationHistory]] = select(AgentConversationHistory)
        count_statement = select(func.count()).select_from(AgentConversationHistory)

        if query.session_id is not None:
            statement = statement.where(AgentConversationHistory.session_id == query.session_id)
            count_statement = count_statement.where(AgentConversationHistory.session_id == query.session_id)
        if query.parser_source is not None:
            statement = statement.where(AgentConversationHistory.parser_source == query.parser_source)
            count_statement = count_statement.where(AgentConversationHistory.parser_source == query.parser_source)
        if query.agent_state is not None:
            statement = statement.where(AgentConversationHistory.agent_state == query.agent_state)
            count_statement = count_statement.where(AgentConversationHistory.agent_state == query.agent_state)
        if query.intent is not None:
            statement = statement.where(AgentConversationHistory.intent == query.intent)
            count_statement = count_statement.where(AgentConversationHistory.intent == query.intent)

        statement = (
            statement.order_by(AgentConversationHistory.id.desc())
            .offset(query.offset)
            .limit(query.limit)
        )
        result = await self._session.execute(statement)
        count_result = await self._session.execute(count_statement)
        return list(result.scalars().all()), int(count_result.scalar_one())

    async def export_logs(
        self,
        query: ConversationHistoryQuery,
    ) -> list[AgentConversationHistory]:
        statement: Select[tuple[AgentConversationHistory]] = select(AgentConversationHistory)

        if query.session_id is not None:
            statement = statement.where(AgentConversationHistory.session_id == query.session_id)
        if query.parser_source is not None:
            statement = statement.where(AgentConversationHistory.parser_source == query.parser_source)
        if query.agent_state is not None:
            statement = statement.where(AgentConversationHistory.agent_state == query.agent_state)
        if query.intent is not None:
            statement = statement.where(AgentConversationHistory.intent == query.intent)

        statement = statement.order_by(AgentConversationHistory.id.desc())
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def summarize(self) -> dict[str, int]:
        statement = select(
            func.count().label("total"),
            func.coalesce(
                func.sum(case((AgentConversationHistory.agent_state == "confirm", 1), else_=0)),
                0,
            ).label("confirm_count"),
            func.coalesce(
                func.sum(case((AgentConversationHistory.agent_state == "clarify", 1), else_=0)),
                0,
            ).label("clarify_count"),
            func.coalesce(
                func.sum(case((AgentConversationHistory.agent_state == "reply", 1), else_=0)),
                0,
            ).label("reply_count"),
            func.coalesce(
                func.sum(case((AgentConversationHistory.parser_source == "llm", 1), else_=0)),
                0,
            ).label("llm_source_count"),
        ).select_from(AgentConversationHistory)

        result = await self._session.execute(statement)
        row = result.one()
        return {
            "total": int(row.total),
            "confirm_count": int(row.confirm_count),
            "clarify_count": int(row.clarify_count),
            "reply_count": int(row.reply_count),
            "llm_source_count": int(row.llm_source_count),
        }
