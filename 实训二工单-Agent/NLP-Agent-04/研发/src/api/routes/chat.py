"""问答路由"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from src.api.schemas.chat import (
    ChatRequestSchema,
    ChatResponseSchema,
    ChatErrorSchema,
    BatchRequestSchema,
    BatchResponseSchema,
    BatchStatusSchema,
    TableInfoSchema,
)
from src.core.engine.pipeline import NL2SQLPipeline
from src.core.models import ChatRequest
from src.services.llm_service import LLMService
from src.services.db_service import DatabaseService
from src.services.cache_service import CacheService
from src.core.retriever.few_shot import FewShotRetriever
from config import settings

router = APIRouter()

# ── 全局 Pipeline 实例（懒初始化） ──
_pipeline: Optional[NL2SQLPipeline] = None


def _get_pipeline() -> NL2SQLPipeline:
    global _pipeline
    if _pipeline is None:
        llm = LLMService()
        db = DatabaseService()
        cache = CacheService()
        few_shot = FewShotRetriever()
        _pipeline = NL2SQLPipeline(
            llm_service=llm,
            db_service=db,
            cache_service=cache,
            few_shot_retriever=few_shot,
        )
    return _pipeline


@router.post("/chat", response_model=ChatResponseSchema)
async def chat(
    request: ChatRequestSchema,
):
    """单轮问答"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        pipeline = _get_pipeline()
        result = pipeline.run(ChatRequest(
            question=request.question,
            model=request.model or "",
            temperature=request.temperature,
            enable_few_shot=request.enable_few_shot,
            session_id=request.session_id or "",
        ))

        return ChatResponseSchema(
            code=0,
            message="success",
            data={
                "question": result.question,
                "answer": result.answer,
                "sql": result.sql,
                "sql_result": {
                    "columns": result.sql_result.columns if result.sql_result else [],
                    "rows": result.sql_result.rows if result.sql_result else [],
                    "row_count": result.sql_result.row_count if result.sql_result else 0,
                } if result.sql_result else None,
                "tables_used": result.tables_used,
                "latency_ms": round(result.latency_ms, 2),
                "model_used": result.model_used,
                "session_id": request.session_id or "",
            },
        )
    except Exception as e:
        logger.error(f"问答处理失败: {e}")
        return ChatResponseSchema(
            code=1001,
            message="系统处理失败",
            data={"question": request.question, "error_detail": str(e)},
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequestSchema,
):
    """流式问答（SSE）"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    pipeline = _get_pipeline()

    async def event_generator():
        # 先返回问题和 SQL
        chat_req = ChatRequest(
            question=request.question,
            model=request.model or "",
            temperature=request.temperature,
            enable_few_shot=request.enable_few_shot,
            session_id=request.session_id or "",
        )
        result = pipeline.run(chat_req)

        yield {
            "event": "result",
            "data": ChatResponseSchema(
                code=0,
                message="success",
                data={
                    "question": result.question,
                    "answer": result.answer,
                    "sql": result.sql,
                    "tables_used": result.tables_used,
                    "latency_ms": round(result.latency_ms, 2),
                    "model_used": result.model_used,
                },
            ).model_dump_json(ensure_ascii=False),
        }

    return EventSourceResponse(event_generator())


@router.post("/batch", response_model=BatchResponseSchema)
async def batch_chat(
    request: BatchRequestSchema,
):
    """批量问答"""
    # 预留：接收 jsonl 文件，异步执行
    from src.batch.batch_runner import BatchRunner

    runner = BatchRunner(_get_pipeline())
    task = runner.submit(request.questions)

    return BatchResponseSchema(
        code=0,
        message="success",
        data={
            "batch_id": task.batch_id,
            "total": task.total,
            "status": task.status,
        },
    )


@router.get("/batch/{batch_id}", response_model=BatchStatusSchema)
async def batch_status(
    batch_id: str,
):
    """查询批处理状态"""
    from src.batch.batch_runner import BatchRunner

    runner = BatchRunner(_get_pipeline())
    task = runner.get_task(batch_id)
    if not task:
        raise HTTPException(status_code=404, detail="批处理任务不存在")

    return BatchStatusSchema(
        code=0,
        message="success",
        data={
            "batch_id": task.batch_id,
            "total": task.total,
            "completed": task.completed,
            "failed": task.failed,
            "status": task.status,
        },
    )


@router.get("/tables", response_model=list[TableInfoSchema])
async def list_tables(
):
    """获取所有数据表信息"""
    from src.services.db_service import DatabaseService

    db = DatabaseService()
    schemas = db.get_all_table_schemas()

    return [
        TableInfoSchema(
            table_name=s.name,
            description=s.description,
            columns=[{"name": c.name, "type": c.data_type} for c in s.columns],
        )
        for s in schemas
    ]
