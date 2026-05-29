from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.chat.local_llm_provider import LocalLLMProvider
from app.chat.models import ChatCompletionResult
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, log_timed, preview_text

logger = get_logger(__name__)


class LLMProviderUnavailableError(RuntimeError):
    pass


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logger

    @log_timed("llm_complete")
    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> ChatCompletionResult:
        provider_errors: list[str] = []
        emit_runtime_trace(
            self.logger,
            "llm_complete_entered",
            message_count=len(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            last_user_message=preview_text(
                next((item["content"] for item in reversed(messages) if item.get("role") == "user"), ""),
                160,
            ),
        )

        for provider in self._iter_providers():
            try:
                emit_runtime_trace(
                    self.logger,
                    "llm_provider_selected",
                    provider=provider["name"],
                    model=provider["model"],
                )
                if provider["name"] == "local_transformers":
                    result = await LocalLLMProvider().complete(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    result = await self._request_non_stream(
                        base_url=str(provider["base_url"]),
                        api_key=str(provider["api_key"]),
                        model=str(provider["model"]),
                        timeout=int(provider["timeout"]),
                        retries=int(provider["retries"]),
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                result.degraded_to_online_api = bool(provider["degraded_to_online_api"])
                emit_runtime_trace(
                    self.logger,
                    "llm_complete_finished",
                    provider=provider["name"],
                    model=result.model,
                    tokens_used=result.tokens_used,
                    degraded=result.degraded_to_online_api,
                    response_preview=preview_text(result.content, 160),
                )
                return result
            except Exception as exc:
                provider_errors.append(f"{provider['name']}: {exc}")
                logger.warning("llm_provider_failed", provider=provider["name"], error=str(exc))

        raise LLMProviderUnavailableError(self._build_unavailable_message(provider_errors))

    @log_timed("llm_stream")
    async def stream(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[dict[str, str], None]:
        provider_errors: list[str] = []
        emit_runtime_trace(
            self.logger,
            "llm_stream_entered",
            message_count=len(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        for provider in self._iter_providers():
            try:
                emit_runtime_trace(
                    self.logger,
                    "llm_stream_provider_selected",
                    provider=provider["name"],
                    model=provider["model"],
                )
                if provider["name"] == "local_transformers":
                    async for event in LocalLLMProvider().stream(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        yield event
                    return

                async for event in self._request_stream(
                    base_url=str(provider["base_url"]),
                    api_key=str(provider["api_key"]),
                    model=str(provider["model"]),
                    timeout=int(provider["timeout"]),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    degraded=bool(provider["degraded_to_online_api"]),
                ):
                    yield event
                return
            except Exception as exc:
                provider_errors.append(f"{provider['name']}: {exc}")
                logger.warning("llm_stream_provider_failed", provider=provider["name"], error=str(exc))

        raise LLMProviderUnavailableError(self._build_unavailable_message(provider_errors))

    async def _request_non_stream(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int,
        retries: int,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResult:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = self._build_headers(api_key)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        emit_runtime_trace(
            self.logger,
            "llm_http_request_prepared",
            url=url,
            model=model,
            message_count=len(messages),
            last_user_message=preview_text(
                next((item["content"] for item in reversed(messages) if item.get("role") == "user"), ""),
                160,
            ),
        )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max(1, retries)),
            wait=wait_fixed(1),
            retry=retry_if_exception_type((httpx.HTTPError, RuntimeError)),
            reraise=True,
        ):
            with attempt:
                emit_runtime_trace(
                    self.logger,
                    "llm_http_request_attempt",
                    url=url,
                    model=model,
                )
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if not content:
                        raise RuntimeError("LLM empty response.")

                    usage = data.get("usage", {})
                    emit_runtime_trace(
                        self.logger,
                        "llm_http_response_received",
                        model=model,
                        content_preview=preview_text(content, 160),
                    )
                    return ChatCompletionResult(
                        content=content,
                        model=model,
                        tokens_used=int(usage.get("total_tokens", 0)),
                        degraded_to_online_api=False,
                        raw_response=data,
                    )

        raise RuntimeError("LLM completion exhausted retries.")

    async def _request_stream(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        degraded: bool,
    ) -> AsyncGenerator[dict[str, str], None]:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = self._build_headers(api_key)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        emit_runtime_trace(
            self.logger,
            "llm_stream_request_prepared",
            url=url,
            model=model,
            message_count=len(messages),
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        emit_runtime_trace(self.logger, "llm_stream_completed", model=model, degraded=degraded)
                        yield {
                            "event": "end",
                            "data": json.dumps({"model": model, "degraded": degraded}, ensure_ascii=False),
                        }
                        break

                    payload_json = json.loads(data)
                    delta = (
                        payload_json.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        yield {
                            "event": "delta",
                            "data": json.dumps(
                                {"content": delta, "model": model, "degraded": degraded},
                                ensure_ascii=False,
                            ),
                        }

    def _build_headers(self, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key and not self._is_placeholder_api_key(api_key):
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _iter_providers(self) -> list[dict[str, str | int | bool]]:
        providers: list[dict[str, str | int | bool]] = []

        if self.settings.vllm_base_url and self.settings.vllm_model:
            providers.append(
                {
                    "name": "vllm",
                    "base_url": self.settings.vllm_base_url,
                    "api_key": self.settings.vllm_api_key,
                    "model": self.settings.vllm_model,
                    "timeout": self.settings.vllm_timeout_seconds,
                    "retries": self.settings.vllm_max_retries,
                    "degraded_to_online_api": False,
                }
            )

        local_provider = LocalLLMProvider()
        if local_provider.is_enabled():
            providers.append(
                {
                    "name": "local_transformers",
                    "base_url": "",
                    "api_key": "",
                    "model": local_provider.model_name,
                    "timeout": 0,
                    "retries": 1,
                    "degraded_to_online_api": False,
                }
            )

        if (
            self.settings.siliconflow_base_url
            and self.settings.siliconflow_model
            and not self._is_placeholder_api_key(self.settings.siliconflow_api_key)
        ):
            providers.append(
                {
                    "name": "siliconflow",
                    "base_url": self.settings.siliconflow_base_url,
                    "api_key": self.settings.siliconflow_api_key,
                    "model": self.settings.siliconflow_model,
                    "timeout": self.settings.siliconflow_timeout_seconds,
                    "retries": self.settings.siliconflow_max_retries,
                    "degraded_to_online_api": True,
                }
            )

        return providers

    def _is_placeholder_api_key(self, api_key: str) -> bool:
        normalized = api_key.strip()
        if not normalized:
            return True
        lowered = normalized.lower()
        return lowered == "empty" or lowered.startswith("replace-with-")

    def _build_unavailable_message(self, provider_errors: list[str]) -> str:
        if not provider_errors:
            return (
                "No available LLM backend. Start the local vLLM service, enable LOCAL_LLM_* "
                "fallback, or configure a valid SILICONFLOW_API_KEY in .env."
            )
        return (
            "No available LLM backend. Start the local vLLM service, enable LOCAL_LLM_* "
            f"fallback, or configure a valid SILICONFLOW_API_KEY in .env. Details: {' | '.join(provider_errors)}"
        )
