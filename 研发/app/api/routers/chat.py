from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse

from app.api.dependencies import get_current_user_id, get_db_session, get_role_service, require_user_match
from app.api.schemas import ChatRequest, ChatResponse, ClearChatRequest, ClearChatResponse, ContextSourceSchema, Envelope
from app.chat.cache_service import ChatCacheService
from app.chat.context_builder import ContextBuilder
from app.chat.llm_client import LLMClient, LLMProviderUnavailableError
from app.chat.memory_service import MemoryService
from app.chat.rate_limiter import RateLimitExceededError, RedisLeakyBucketRateLimiter
from app.chat.role_guard import RoleGuard
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, preview_text
from app.core.request_context import get_request_id
from app.core.response import success_response
from app.db.redis_client import (
    chat_recent_key,
    chat_session_key,
    get_redis,
    memory_summary_key,
    query_cache_pattern,
)
from app.services.role_service import RoleService

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)

MIN_VISIBLE_CONTEXT_SCORE = 0.5
MAX_VISIBLE_CONTEXT_SOURCES = 5


def _filter_context_sources(sources: list[ContextSourceSchema]) -> list[ContextSourceSchema]:
    return [source for source in sources if source.score >= MIN_VISIBLE_CONTEXT_SCORE][:MAX_VISIBLE_CONTEXT_SOURCES]


@router.post("", response_model=Envelope[ChatResponse])
async def chat(
    payload: ChatRequest,
    db_session: AsyncSession = Depends(get_db_session),
    current_user_id: str | None = Depends(get_current_user_id),
    role_service: RoleService = Depends(get_role_service),
):
    emit_runtime_trace(
        logger,
        "chat_api_entered",
        user_id=payload.user_id,
        role_id=payload.role_id,
        role_name=payload.role_name,
        stream=payload.stream,
        top_k=payload.top_k,
        temperature=payload.temperature,
        query=payload.query,
    )

    require_user_match(payload.user_id, current_user_id)
    settings = get_settings()
    limiter = RedisLeakyBucketRateLimiter()
    cache_service = ChatCacheService()

    try:
        role = await role_service.resolve_role(
            db_session,
            user_id=payload.user_id,
            role_id=payload.role_id,
            role_name=payload.role_name,
        )
        emit_runtime_trace(
            logger,
            "chat_api_role_resolved",
            user_id=payload.user_id,
            role_id=role.role_id,
            role_name=role.name,
            role_category=role.category,
        )
        await limiter.check(user_id=payload.user_id, role_id=role.role_id)
        emit_runtime_trace(
            logger,
            "chat_api_rate_limit_passed",
            user_id=payload.user_id,
            role_id=role.role_id,
        )
    except RateLimitExceededError as exc:
        emit_runtime_trace(logger, "chat_api_rate_limit_blocked", error=str(exc))
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        emit_runtime_trace(logger, "chat_api_role_resolution_failed", error=str(exc))
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    request_id = get_request_id() or uuid.uuid4().hex
    session_id = payload.session_id or uuid.uuid4().hex
    role_guard = RoleGuard()
    system_prompt = role_guard.build_system_prompt(
        role_name=role.name,
        role_category=role.category,
        system_prompt=role.system_prompt,
    )
    started_at = time.perf_counter()

    emit_runtime_trace(
        logger,
        "chat_api_request_initialized",
        request_id=request_id,
        session_id=session_id,
        system_prompt_preview=preview_text(system_prompt, 160),
    )

    if not payload.stream:
        cached_response = await cache_service.get_cached_response(
            user_id=payload.user_id,
            role_id=role.role_id,
            session_id=session_id,
            query=payload.query,
        )
        if cached_response is not None:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            emit_runtime_trace(
                logger,
                "chat_api_cache_hit",
                request_id=request_id,
                session_id=session_id,
                latency_ms=latency_ms,
            )
            return success_response(
                cached_response.model_copy(
                    update={
                        "request_id": request_id,
                        "session_id": session_id,
                        "latency_ms": latency_ms,
                        "context_sources": _filter_context_sources(cached_response.context_sources),
                    }
                )
            )
        emit_runtime_trace(
            logger,
            "chat_api_cache_miss",
            request_id=request_id,
            session_id=session_id,
            query=payload.query,
        )

    context_builder = ContextBuilder()
    llm_client = LLMClient()
    memory_service = MemoryService()

    emit_runtime_trace(
        logger,
        "chat_api_context_build_started",
        request_id=request_id,
        session_id=session_id,
    )
    built_context = await context_builder.build(
        db_session=db_session,
        user_id=payload.user_id,
        role_id=role.role_id,
        role_name=role.name,
        role_category=role.category,
        system_prompt=system_prompt,
        query=payload.query,
        session_id=session_id,
        top_k=payload.top_k,
    )
    emit_runtime_trace(
        logger,
        "chat_api_context_build_finished",
        request_id=request_id,
        session_id=session_id,
        rewritten_query=built_context.rewritten_query,
        context_source_count=len(built_context.context_sources),
        first_context_preview=preview_text(built_context.context_sources[0].text, 160) if built_context.context_sources else "",
    )

    if payload.stream:
        async def event_generator() -> AsyncGenerator[dict[str, str], None]:
            buffer: list[str] = []
            stream_model = ""
            stream_degraded = False

            yield {"event": "start", "data": json.dumps({"request_id": request_id, "session_id": session_id}, ensure_ascii=False)}
            for source in built_context.context_sources:
                yield {
                    "event": "source",
                    "data": json.dumps(
                        {
                            "doc_id": source.doc_id,
                            "chunk_id": source.chunk_id,
                            "source": source.source,
                            "score": source.score,
                        },
                        ensure_ascii=False,
                    ),
                }

            try:
                emit_runtime_trace(
                    logger,
                    "chat_api_stream_llm_started",
                    request_id=request_id,
                    session_id=session_id,
                )
                async for event in llm_client.stream(
                    messages=built_context.messages,
                    temperature=payload.temperature,
                ):
                    if event["event"] == "delta":
                        payload_data = json.loads(event["data"])
                        content = payload_data.get("content", "")
                        stream_model = payload_data.get("model", stream_model)
                        stream_degraded = bool(payload_data.get("degraded", stream_degraded))
                        buffer.append(content)
                    elif event["event"] == "end":
                        payload_data = json.loads(event["data"])
                        stream_model = payload_data.get("model", stream_model)
                        stream_degraded = bool(payload_data.get("degraded", stream_degraded))
                    yield event
            except LLMProviderUnavailableError as exc:
                emit_runtime_trace(
                    logger,
                    "chat_api_stream_llm_failed",
                    request_id=request_id,
                    error=str(exc),
                )
                yield {
                    "event": "error",
                    "data": json.dumps({"message": str(exc), "request_id": request_id}, ensure_ascii=False),
                }
                return

            final_response = role_guard.validate_and_postprocess(
                response_text="".join(buffer),
                role_category=role.category,
                role_name=role.name,
            )
            emit_runtime_trace(
                logger,
                "chat_api_stream_postprocess_finished",
                request_id=request_id,
                response_preview=preview_text(final_response, 160),
            )

            emit_runtime_trace(logger, "chat_api_stream_persist_started", request_id=request_id)
            await _persist_chat(
                user_id=payload.user_id,
                role_id=role.role_id,
                session_id=session_id,
                query=payload.query,
                response=final_response,
                tokens_used=0,
                db_session=db_session,
            )
            emit_runtime_trace(logger, "chat_api_stream_persist_finished", request_id=request_id)

            emit_runtime_trace(logger, "chat_api_stream_memory_update_started", request_id=request_id)
            await memory_service.update_summary(
                user_id=payload.user_id,
                role_id=role.role_id,
                session_id=session_id,
                query=payload.query,
                response=final_response,
            )
            emit_runtime_trace(logger, "chat_api_stream_memory_update_finished", request_id=request_id)

            await cache_service.set_cached_response(
                user_id=payload.user_id,
                role_id=role.role_id,
                session_id=session_id,
                query=payload.query,
                response=ChatResponse(
                    request_id=request_id,
                    role_id=role.role_id,
                    role_name=role.name,
                    session_id=session_id,
                    response=final_response,
                    context_sources=[
                        ContextSourceSchema(
                            doc_id=item.doc_id,
                            chunk_id=item.chunk_id,
                            source=item.source,
                            score=item.score,
                        )
                        for item in built_context.context_sources
                    ],
                    tokens_used=0,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    model=stream_model or settings.vllm_model,
                    degraded_to_online_api=stream_degraded,
                    rewritten_query=built_context.rewritten_query,
                ),
            )
            emit_runtime_trace(logger, "chat_api_stream_cache_set", request_id=request_id)

        return EventSourceResponse(event_generator(), ping=settings.sse_heartbeat_seconds)

    try:
        emit_runtime_trace(
            logger,
            "chat_api_llm_started",
            request_id=request_id,
            session_id=session_id,
            rewritten_query=built_context.rewritten_query,
        )
        result = await llm_client.complete(
            messages=built_context.messages,
            temperature=payload.temperature,
        )
    except LLMProviderUnavailableError as exc:
        emit_runtime_trace(
            logger,
            "chat_api_llm_failed",
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    emit_runtime_trace(
        logger,
        "chat_api_llm_finished",
        request_id=request_id,
        model=result.model,
        tokens_used=result.tokens_used,
        degraded=result.degraded_to_online_api,
        response_preview=preview_text(result.content, 160),
    )

    final_response = role_guard.validate_and_postprocess(
        response_text=result.content,
        role_category=role.category,
        role_name=role.name,
    )
    emit_runtime_trace(
        logger,
        "chat_api_postprocess_finished",
        request_id=request_id,
        response_preview=preview_text(final_response, 160),
    )

    emit_runtime_trace(logger, "chat_api_persist_started", request_id=request_id)
    await _persist_chat(
        user_id=payload.user_id,
        role_id=role.role_id,
        session_id=session_id,
        query=payload.query,
        response=final_response,
        tokens_used=result.tokens_used,
        db_session=db_session,
    )
    emit_runtime_trace(logger, "chat_api_persist_finished", request_id=request_id)

    emit_runtime_trace(logger, "chat_api_memory_update_started", request_id=request_id)
    await memory_service.update_summary(
        user_id=payload.user_id,
        role_id=role.role_id,
        session_id=session_id,
        query=payload.query,
        response=final_response,
    )
    emit_runtime_trace(logger, "chat_api_memory_update_finished", request_id=request_id)

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    chat_response = ChatResponse(
        request_id=request_id,
        role_id=role.role_id,
        role_name=role.name,
        session_id=session_id,
        response=final_response,
        context_sources=[
            ContextSourceSchema(
                doc_id=item.doc_id,
                chunk_id=item.chunk_id,
                source=item.source,
                score=item.score,
            )
            for item in built_context.context_sources
        ],
        tokens_used=result.tokens_used,
        latency_ms=latency_ms,
        model=result.model,
        degraded_to_online_api=result.degraded_to_online_api,
        rewritten_query=built_context.rewritten_query,
    )
    await cache_service.set_cached_response(
        user_id=payload.user_id,
        role_id=role.role_id,
        session_id=session_id,
        query=payload.query,
        response=chat_response,
    )
    emit_runtime_trace(
        logger,
        "chat_api_completed",
        request_id=request_id,
        session_id=session_id,
        latency_ms=latency_ms,
        context_source_count=len(chat_response.context_sources),
    )

    return success_response(chat_response)


@router.post("/clear", response_model=Envelope[ClearChatResponse])
async def clear_chat(
    payload: ClearChatRequest,
    current_user_id: str | None = Depends(get_current_user_id),
):
    require_user_match(payload.user_id, current_user_id)
    redis = get_redis()

    emit_runtime_trace(
        logger,
        "chat_clear_entered",
        user_id=payload.user_id,
        role_id=payload.role_id,
        session_id=payload.session_id,
    )

    keys = [
        chat_recent_key(payload.user_id, payload.role_id, payload.session_id),
        chat_session_key(payload.user_id, payload.role_id, payload.session_id),
        memory_summary_key(payload.user_id, payload.role_id, payload.session_id),
    ]
    cache_keys = [
        key
        async for key in redis.scan_iter(
            match=query_cache_pattern(payload.user_id, payload.role_id, payload.session_id),
            count=100,
        )
    ]
    all_keys = keys + cache_keys
    cleared_count = await redis.delete(*all_keys) if all_keys else 0

    emit_runtime_trace(
        logger,
        "chat_clear_completed",
        user_id=payload.user_id,
        role_id=payload.role_id,
        cleared_count=cleared_count,
    )
    return success_response(
        ClearChatResponse(
            success=cleared_count >= 0,
            cleared_keys=all_keys,
            session_id=payload.session_id,
        )
    )


async def _persist_chat(
    *,
    user_id: str,
    role_id: str,
    query: str,
    response: str,
    tokens_used: int,
    db_session: AsyncSession,
    session_id: str,
) -> None:
    settings = get_settings()
    redis = get_redis()

    emit_runtime_trace(
        logger,
        "chat_persist_entered",
        user_id=user_id,
        role_id=role_id,
        session_id=session_id,
        query=query,
        response_preview=preview_text(response, 160),
        tokens_used=tokens_used,
    )

    recent_key = chat_recent_key(user_id, role_id, session_id)
    session_key = chat_session_key(user_id, role_id, session_id)

    await redis.rpush(recent_key, json.dumps({"role": "user", "content": query}, ensure_ascii=False))
    await redis.rpush(recent_key, json.dumps({"role": "assistant", "content": response}, ensure_ascii=False))
    await redis.ltrim(recent_key, -settings.chat_recent_rounds * 2, -1)
    await redis.expire(recent_key, settings.redis_chat_recent_ttl_seconds)
    await redis.set(session_key, session_id, ex=settings.redis_chat_session_ttl_seconds)

    emit_runtime_trace(
        logger,
        "chat_persist_redis_finished",
        recent_key=recent_key,
        session_key=session_key,
    )

    insert_stmt = text(
        """
        INSERT INTO conversations (user_id, role_id, query, response, tokens_used, timestamp)
        VALUES (:user_id, :role_id, :query, :response, :tokens_used, NOW())
        """
    )
    await db_session.execute(
        insert_stmt,
        {
            "user_id": user_id,
            "role_id": role_id,
            "query": query,
            "response": response,
            "tokens_used": tokens_used,
        },
    )

    mapping_stmt = text(
        """
        INSERT INTO user_role_mapping (user_id, role_id, last_used_at, total_interactions)
        VALUES (:user_id, :role_id, NOW(), 1)
        ON DUPLICATE KEY UPDATE
          last_used_at = NOW(),
          total_interactions = total_interactions + 1
        """
    )
    await db_session.execute(mapping_stmt, {"user_id": user_id, "role_id": role_id})
    await db_session.commit()
    logger.info("chat_persisted", user_id=user_id, role_id=role_id)
    emit_runtime_trace(
        logger,
        "chat_persist_completed",
        user_id=user_id,
        role_id=role_id,
        session_id=session_id,
    )
