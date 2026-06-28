import json

from structlog import get_logger

from app.agents.llm_response_prompts import LLM_SUGGESTIONS_SYSTEM_PROMPT
from app.core.config import settings
from app.llm import LLMClient, get_llm_client

logger = get_logger()


class AgentLLMSuggestionWriter:
    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_llm_client()

    async def generate_suggestions(
        self,
        *,
        intent: str,
        agent_state: str,
        user_input: str,
        missing_fields: list[str],
        fallback_suggestions: list[str],
    ) -> list[str]:
        if not settings.llm_reply_enabled or self._client is None:
            return fallback_suggestions

        user_prompt = json.dumps(
            {
                "intent": intent,
                "agent_state": agent_state,
                "user_input": user_input,
                "missing_fields": missing_fields,
                "fallback_suggestions": fallback_suggestions,
            },
            ensure_ascii=False,
        )

        try:
            content = await self._client.chat_text(
                system_prompt=LLM_SUGGESTIONS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                use_json_mode=True,
            )
            payload = json.loads(content)
            suggestions = payload.get("suggestions")
            if not isinstance(suggestions, list):
                return fallback_suggestions
            cleaned = [str(item).strip() for item in suggestions if str(item).strip()]
            return cleaned[:3] or fallback_suggestions
        except Exception as exc:  # pragma: no cover
            logger.warning("llm_suggestions_generation_failed", error=str(exc))
            return fallback_suggestions
