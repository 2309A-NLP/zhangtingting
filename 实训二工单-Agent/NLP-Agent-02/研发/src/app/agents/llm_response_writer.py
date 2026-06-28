import json
from typing import Any

from structlog import get_logger

from app.agents.llm_response_prompts import LLM_REPLY_SYSTEM_PROMPT
from app.core.config import settings
from app.llm import LLMClient, get_llm_client

logger = get_logger()


class AgentLLMResponseWriter:
    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_llm_client()

    async def rewrite_message(
        self,
        *,
        intent: str,
        agent_state: str,
        tool_name: str | None,
        tool_arguments: dict[str, Any],
        execution_result: Any,
        fallback_message: str,
    ) -> str:
        if not settings.llm_reply_enabled or self._client is None:
            return fallback_message

        user_prompt = json.dumps(
            {
                "intent": intent,
                "agent_state": agent_state,
                "tool_name": tool_name,
                "tool_arguments": tool_arguments,
                "execution_result": execution_result,
                "fallback_message": fallback_message,
            },
            ensure_ascii=False,
            default=self._json_default,
        )

        try:
            content = await self._client.chat_text(
                system_prompt=LLM_REPLY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            rewritten = content.strip()
            if not rewritten:
                return fallback_message
            return rewritten
        except Exception as exc:  # pragma: no cover
            logger.warning("llm_reply_rewrite_failed", error=str(exc))
            return fallback_message

    async def rewrite_reply(
        self,
        *,
        intent: str,
        agent_state: str,
        tool_name: str | None,
        tool_arguments: dict[str, Any],
        execution_result: Any,
        fallback_message: str,
    ) -> str:
        return await self.rewrite_message(
            intent=intent,
            agent_state=agent_state,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            execution_result=execution_result,
            fallback_message=fallback_message,
        )

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if hasattr(value, "value"):
            return value.value
        return str(value)
