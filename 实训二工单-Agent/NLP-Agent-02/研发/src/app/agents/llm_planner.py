import json
from datetime import date
from typing import Any

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from app.agents.llm_plan_prompts import LLM_PLAN_SYSTEM_PROMPT
from app.agents.llm_plan_schemas import LLMPlanResponse
from app.core.config import settings
from app.llm import LLMClient, get_llm_client
from app.services.llm_audit_service import LLMAuditLogService

logger = get_logger()


class AgentLLMPlanner:
    def __init__(
        self,
        client: LLMClient | None = None,
        audit_service: LLMAuditLogService | None = None,
    ) -> None:
        self._client = client or get_llm_client()
        self._audit_service = audit_service

    async def plan(
        self,
        *,
        session_id: str,
        user_input: str,
        context: dict[str, Any],
    ) -> LLMPlanResponse | None:
        if not settings.llm_enabled or self._client is None:
            return None

        user_prompt = json.dumps(
            {
                "today": date.today().isoformat(),
                "session_id": session_id,
                "user_input": user_input,
                "context": context,
            },
            ensure_ascii=False,
        )

        raw_content: str | None = None
        payload: dict[str, Any] | None = None
        try:
            raw_content = await self._client.chat_text(
                system_prompt=LLM_PLAN_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                use_json_mode=True,
            )
            payload = json.loads(raw_content)
            parsed = LLMPlanResponse.model_validate(payload)
            await self._record_audit(
                session_id=session_id,
                user_input=user_input,
                success=True,
                request_payload=json.loads(user_prompt),
                raw_response_text=raw_content,
                parsed_response=parsed.model_dump(),
                error_message=None,
            )
            return parsed
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.warning("llm_plan_invalid_response", error=str(exc))
            await self._record_audit(
                session_id=session_id,
                user_input=user_input,
                success=False,
                request_payload=json.loads(user_prompt),
                raw_response_text=raw_content,
                parsed_response=payload,
                error_message=str(exc),
            )
            return None
        except Exception as exc:  # pragma: no cover
            logger.warning("llm_plan_failed", error=str(exc))
            await self._record_audit(
                session_id=session_id,
                user_input=user_input,
                success=False,
                request_payload=json.loads(user_prompt),
                raw_response_text=raw_content,
                parsed_response=payload,
                error_message=str(exc),
            )
            return None

    async def _record_audit(
        self,
        *,
        session_id: str,
        user_input: str,
        success: bool,
        request_payload: dict[str, Any] | None,
        raw_response_text: str | None,
        parsed_response: dict[str, Any] | None,
        error_message: str | None,
    ) -> None:
        if self._audit_service is None:
            return
        provider = self._client.provider_name if self._client is not None else None
        model_name = self._client.model_name if self._client is not None else None
        await self._audit_service.record(
            session_id=session_id,
            user_input=user_input,
            parser_stage="plan",
            success=success,
            provider=provider,
            model_name=model_name,
            request_payload=request_payload,
            raw_response_text=raw_response_text,
            parsed_response=parsed_response,
            error_message=error_message,
        )

    @classmethod
    def from_session(cls, session: AsyncSession, client: LLMClient | None = None) -> "AgentLLMPlanner":
        return cls(
            client=client,
            audit_service=LLMAuditLogService.from_session(session),
        )
