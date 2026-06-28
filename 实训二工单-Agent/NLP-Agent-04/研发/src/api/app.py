"""FastAPI 主应用"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from loguru import logger

from config import settings
from src.api.routes import chat, health
from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.request_log import RequestLogMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info(f"基金数据问答智能体系统启动 — 端口 {settings.API_PORT}")
    logger.info(f"LLM 模型: {settings.LLM_MODEL_NAME} | Provider: {settings.LLM_PROVIDER}")
    logger.info(f"数据库: {settings.DB_PATH}")
    yield
    logger.info("系统关闭")


app = FastAPI(
    title="基金数据问答智能体系统 API",
    description="基于大语言模型的基金数据 NL2SQL 问答系统",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Swagger 安全认证配置 ──
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="基金数据问答智能体系统 API",
        version="1.0.0",
        description="基于大语言模型的基金数据 NL2SQL 问答系统",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "输入 API Key（如 dev-key）",
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", [{"ApiKeyAuth": []}])
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 中间件 ──
app.add_middleware(RequestLogMiddleware)
if settings.API_KEY_ENABLED:
    app.add_middleware(AuthMiddleware)

# ── 路由 ──
app.include_router(health.router, tags=["系统"])
app.include_router(chat.router, prefix="/api/v1", tags=["问答"])
