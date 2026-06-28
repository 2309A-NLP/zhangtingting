import json
from typing import Annotated, Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.repositories.llm_audit_repository import LLMAuditLogRepository
from app.schemas.llm_audit import (
    LLMAuditLogCreate,
    LLMAuditLogList,
    LLMAuditLogQuery,
    LLMAuditLogRead,
)


class LLMAuditLogService:
    def __init__(self, repository: LLMAuditLogRepository) -> None:
        self._repository = repository

    @classmethod
    def from_session(cls, session: AsyncSession) -> "LLMAuditLogService":
        return cls(LLMAuditLogRepository(session))

    async def record(
        self,
        *,
        session_id: str,
        user_input: str,
        parser_stage: str,
        success: bool,
        provider: str | None = None,
        model_name: str | None = None,
        request_payload: dict[str, Any] | None = None,
        raw_response_text: str | None = None,
        parsed_response: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> LLMAuditLogRead:
        payload = LLMAuditLogCreate(
            session_id=session_id,
            user_input=user_input,
            parser_stage=parser_stage,
            provider=provider,
            model_name=model_name,
            success=success,
            request_payload_json=self._dump_json(request_payload),
            raw_response_text=raw_response_text,
            parsed_response_json=self._dump_json(parsed_response),
            error_message=error_message,
        )
        record = await self._repository.create(payload)
        return LLMAuditLogRead.model_validate(record)

    async def list_by_session_id(self, session_id: str) -> list[LLMAuditLogRead]:
        records = await self._repository.list_by_session_id(session_id)
        return [LLMAuditLogRead.model_validate(item) for item in records]

    async def list_logs(
        self,
        *,
        session_id: str | None = None,
        parser_stage: str | None = None,
        success: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> LLMAuditLogList:
        query = LLMAuditLogQuery(
            session_id=session_id,
            parser_stage=parser_stage,
            success=success,
            limit=limit,
            offset=offset,
        )
        records, total = await self._repository.list_logs(query)
        items = [LLMAuditLogRead.model_validate(item) for item in records]
        return LLMAuditLogList(total=total, limit=limit, offset=offset, items=items)

    async def export_logs(
        self,
        *,
        session_id: str | None = None,
        parser_stage: str | None = None,
        success: bool | None = None,
    ) -> list[LLMAuditLogRead]:
        query = LLMAuditLogQuery(
            session_id=session_id,
            parser_stage=parser_stage,
            success=success,
        )
        records = await self._repository.export_logs(query)
        return [LLMAuditLogRead.model_validate(item) for item in records]

    @staticmethod
    def _dump_json(payload: dict[str, Any] | None) -> str | None:
        if payload is None:
            return None
        return json.dumps(payload, ensure_ascii=False)


async def get_llm_audit_log_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> LLMAuditLogService:
    return LLMAuditLogService.from_session(session)
