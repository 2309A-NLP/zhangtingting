from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
# FastAPI 的生命周期管理器（Lifespan Manager），定义了不同服务启动和关闭时的行为
from fastapi import FastAPI

from app.core.config import settings
from app.core.database import close_database, initialize_database
from app.scheduler.runner import create_scheduler, create_worker_scheduler


@asynccontextmanager
async def api_lifespan(_: FastAPI) -> AsyncIterator[None]:
    if settings.auto_create_tables:
        await initialize_database()
    try:
        yield
    finally:
        await close_database()


@asynccontextmanager
async def scheduler_lifespan(_: FastAPI) -> AsyncIterator[None]:
    if settings.auto_create_tables:
        await initialize_database()
    scheduler = create_scheduler()
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        await close_database()
# 执行的调度器任务：
# scan_due_reminders（每 1 分钟）
# clear_expired_conversations（每 10 分钟）

@asynccontextmanager
async def worker_lifespan(_: FastAPI) -> AsyncIterator[None]:
    if settings.auto_create_tables:
        await initialize_database()
    scheduler = create_worker_scheduler()
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        await close_database()
# 执行的调度器任务：
# process_reminder_delivery_tasks（每 10 秒）