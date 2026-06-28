"""健康检查路由"""

from __future__ import annotations

import time

from fastapi import APIRouter

from config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """系统健康检查"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": int(time.time()),
        "llm_model": settings.LLM_MODEL_NAME,
        "llm_provider": settings.LLM_PROVIDER,
        "db_path": settings.DB_PATH,
    }


@router.get("/")
async def root():
    """根路径"""
    return {
        "service": "基金数据问答智能体系统",
        "version": "1.0.0",
        "docs": "/docs",
    }
