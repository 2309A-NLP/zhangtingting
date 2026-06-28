import json
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
from structlog import get_logger

from app.core.config import settings

logger = get_logger()


class LLMClient(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        use_json_mode: bool = False,
    ) -> str:
        raise NotImplementedError

    async def chat_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        content = await self.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_json_mode=True,
        )
        parsed = self._parse_json_content(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON must be an object")
        return parsed

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match is None:
                raise ValueError("LLM response does not contain valid JSON") from None
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON must be an object")
        return parsed


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self._model

    async def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        use_json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            if settings.llm_debug_logging:
                logger.info(
                    "llm_http_request_started",
                    model=self._model,
                    base_url=self._base_url,
                    use_json_mode=use_json_mode,
                    max_tokens=self._max_tokens,
                )
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            response_json = response.json()

        content = response_json["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("LLM response content must be a string")
        if settings.llm_debug_logging:
            logger.info(
                "llm_http_request_completed",
                model=self._model,
                use_json_mode=use_json_mode,
                content_length=len(content),
                preview=content[:300],
            )
        return content


def get_llm_client() -> LLMClient | None:
    if not settings.llm_enabled:
        return None
    if not settings.llm_api_key or not settings.llm_model:
        return None
    return OpenAICompatibleLLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
