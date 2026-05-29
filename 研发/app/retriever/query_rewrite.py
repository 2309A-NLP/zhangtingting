from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.chat.local_llm_provider import LocalLLMProvider
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, log_timed, preview_text
from app.retriever.models import RewriteResult

logger = get_logger(__name__)

REWRITE_SYSTEM_PROMPT = """
You are a RAG query rewriter.
Goals:
1. Preserve the user's original intent.
2. Use conversation history to resolve references, time expressions, and missing subjects.
3. Output a query optimized for retrieval, not a final answer.
4. If the original query is already clear, return it unchanged.

Return JSON only:
{"rewritten_query":"...","reason":"..."}
"""


class QueryRewriter:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logger

    @log_timed("query_rewrite")
    async def rewrite(self, query: str, history: list[dict[str, str]] | None = None) -> RewriteResult:
        if not self.settings.retrieval_enable_query_rewrite:
            emit_runtime_trace(self.logger, "query_rewrite_skipped", reason="rewrite_disabled", query=query)
            return RewriteResult(original_query=query, rewritten_query=query, reason="rewrite_disabled")

        emit_runtime_trace(
            self.logger,
            "query_rewrite_entered",
            query=query,
            history_count=len(history or []),
        )
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_prompt(query=query, history=history or []),
            },
        ]

        content = await self._call_with_fallback(messages)
        if not content:
            logger.info("query_rewrite_skipped", original_query=query, reason="provider_unavailable")
            emit_runtime_trace(
                self.logger,
                "query_rewrite_provider_unavailable",
                query=query,
            )
            return RewriteResult(original_query=query, rewritten_query=query, reason="provider_unavailable")

        rewritten_query, reason = self._parse_response(content, fallback=query)
        logger.info("query_rewritten", original_query=query, rewritten_query=rewritten_query, reason=reason)
        emit_runtime_trace(
            self.logger,
            "query_rewrite_completed",
            original_query=query,
            rewritten_query=rewritten_query,
            reason=reason,
        )
        return RewriteResult(original_query=query, rewritten_query=rewritten_query, reason=reason)

    def _build_prompt(self, query: str, history: list[dict[str, str]]) -> str:
        history_lines = []
        for item in history[-5:]:
            role = item.get("role", "user")
            content = item.get("content", "").strip()
            if content:
                history_lines.append(f"{role}: {content}")

        joined_history = "\n".join(history_lines) if history_lines else "No history."
        return f"Conversation history:\n{joined_history}\n\nCurrent query:\n{query}"

    async def _call_with_fallback(self, messages: list[dict[str, str]]) -> str | None:
        provider_errors: list[str] = []
        local_provider = LocalLLMProvider()
        providers = [
            {
                "name": "vllm",
                "base_url": self.settings.vllm_base_url,
                "api_key": self.settings.vllm_api_key,
                "model": self.settings.vllm_model,
                "timeout": self.settings.vllm_timeout_seconds,
                "retries": self.settings.vllm_max_retries,
                "enabled": bool(self.settings.vllm_base_url and self.settings.vllm_model),
            },
            {
                "name": "local_transformers",
                "base_url": "",
                "api_key": "",
                "model": local_provider.model_name,
                "timeout": 0,
                "retries": 1,
                "enabled": local_provider.is_enabled(),
            },
            {
                "name": "siliconflow",
                "base_url": self.settings.siliconflow_base_url,
                "api_key": self.settings.siliconflow_api_key,
                "model": self.settings.siliconflow_model,
                "timeout": self.settings.siliconflow_timeout_seconds,
                "retries": self.settings.siliconflow_max_retries,
                "enabled": self._is_online_provider_enabled(self.settings.siliconflow_api_key),
            },
        ]

        for provider in providers:
            if not provider["enabled"]:
                provider_errors.append(f"{provider['name']}: not_configured")
                continue

            try:
                emit_runtime_trace(
                    self.logger,
                    "query_rewrite_provider_selected",
                    provider=provider["name"],
                    model=provider["model"],
                )
                if provider["name"] == "local_transformers":
                    result = await local_provider.complete(
                        messages=messages,
                        temperature=0.1,
                        max_tokens=256,
                    )
                    return result.content

                return await self._call_chat_completion(
                    base_url=str(provider["base_url"]),
                    api_key=str(provider["api_key"]),
                    model=str(provider["model"]),
                    timeout=int(provider["timeout"]),
                    retries=int(provider["retries"]),
                    messages=messages,
                )
            except Exception as exc:
                provider_errors.append(f"{provider['name']}: {exc}")
                logger.warning("query_rewrite_provider_failed", provider=provider["name"], error=str(exc))

        logger.warning("query_rewrite_all_providers_failed", errors=provider_errors)
        return None

    async def _call_chat_completion(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int,
        retries: int,
        messages: list[dict[str, str]],
    ) -> str:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "stream": False,
            "max_tokens": 256,
        }

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max(1, retries)),
            wait=wait_fixed(1),
            retry=retry_if_exception_type((httpx.HTTPError, RuntimeError)),
            reraise=True,
        ):
            with attempt:
                emit_runtime_trace(
                    self.logger,
                    "query_rewrite_http_request_attempt",
                    model=model,
                    url=url,
                    prompt_preview=preview_text(messages[-1]["content"], 160),
                )
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if not content:
                        raise RuntimeError("Empty rewrite response.")
                    emit_runtime_trace(
                        self.logger,
                        "query_rewrite_http_response_received",
                        model=model,
                        content_preview=preview_text(content, 160),
                    )
                    return content

        raise RuntimeError("Rewrite request exhausted retries.")

    def _parse_response(self, content: str, fallback: str) -> tuple[str, str]:
        try:
            parsed: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("query_rewrite_invalid_json", content=content)
            return fallback, "invalid_json"

        rewritten = str(parsed.get("rewritten_query", "")).strip() or fallback
        reason = str(parsed.get("reason", "ok")).strip() or "ok"
        return rewritten, reason

    def _is_online_provider_enabled(self, api_key: str) -> bool:
        normalized = api_key.strip()
        if not normalized:
            return False
        lowered = normalized.lower()
        return lowered != "empty" and not lowered.startswith("replace-with-")
