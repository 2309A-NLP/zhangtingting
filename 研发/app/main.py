from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routers.auth import router as auth_router
from app.api.routers.chat import router as chat_router
from app.api.routers.knowledge import router as knowledge_router
from app.api.routers.roles import router as roles_router
from app.api.schemas import Envelope, HealthResponse
from app.chat.local_llm_provider import LocalLLMProvider
from app.chat.vllm_service import VLLMService
from app.core.config import get_settings
from app.core.exception_handlers import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestContextMiddleware
from app.core.response import success_response
from app.db.milvus_client import close_milvus, get_milvus_client, init_milvus
from app.db.mysql_client import (
    close_mysql,
    get_mysql_engine,
    get_mysql_session_factory,
    init_mysql,
)
from app.db.redis_client import close_redis, get_redis, init_redis
from app.knowledge.task_queue import start_knowledge_task_queue, stop_knowledge_task_queue
from app.services.role_service import RoleService

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    vllm_service = VLLMService()

    await init_mysql()

    session_factory = get_mysql_session_factory()
    async with session_factory() as session:
        await RoleService().ensure_preset_roles_seeded(session)

    await init_redis()
    await init_milvus()
    await start_knowledge_task_queue()

    if settings.llm_warmup_on_startup:
        vllm_status, _ = await vllm_service.healthcheck()
        if vllm_status == "ok":
            await vllm_service.warmup()
        else:
            local_cpu_status, _ = LocalLLMProvider().healthcheck()
            if local_cpu_status == "ok":
                logger.info(
                    "llm_warmup_skipped",
                    provider="local_transformers",
                    reason="vllm_unavailable_cpu_fallback_ready",
                )
            else:
                is_ready = await vllm_service.wait_until_ready()
                if is_ready:
                    await vllm_service.warmup()

    logger.info("application_started")

    try:
        yield
    finally:
        await stop_knowledge_task_queue()
        await close_milvus()
        await close_redis()
        await close_mysql()
        logger.info("application_stopped")


app = FastAPI(
    title=settings.app_name,
    debug=settings.app_debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app_cors_origins or ["*"],
    allow_origin_regex=settings.app_cors_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(chat_router, prefix=settings.app_api_prefix)
app.include_router(roles_router, prefix=settings.app_api_prefix)
app.include_router(knowledge_router, prefix=settings.app_api_prefix)
app.include_router(auth_router, prefix=settings.app_api_prefix)


@app.get(
    f"{settings.app_api_prefix}/health",
    response_model=Envelope[HealthResponse],
    tags=["health"],
)
async def health() -> Envelope[HealthResponse]:
    mysql_status = "ok"
    redis_status = "ok"
    milvus_status = "ok"

    try:
        engine = get_mysql_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        mysql_status = "down"

    try:
        await get_redis().ping()
    except Exception:
        redis_status = "down"

    try:
        get_milvus_client().list_collections()
    except Exception:
        milvus_status = "down"

    llm_local_status, _ = await VLLMService().healthcheck()
    if llm_local_status != "ok":
        local_cpu_status, _ = LocalLLMProvider().healthcheck()
        if local_cpu_status == "ok":
            llm_local_status = "cpu_fallback_only"

    services = {
        "mysql": mysql_status,
        "redis": redis_status,
        "milvus": milvus_status,
        "llm_local": llm_local_status,
        "llm_online_fallback": "unknown",
    }
    overall = "ok" if all(value == "ok" for value in services.values()) else "degraded"

    return success_response(
        HealthResponse(
            status=overall,  # type: ignore[arg-type]
            version="1.0.0",
            services=services,
        )
    )
