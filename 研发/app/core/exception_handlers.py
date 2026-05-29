from __future__ import annotations

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.core.response import error_response

logger = get_logger(__name__)

# exc  捕获到的 HTTPException 异常实例
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    payload = error_response(
        code=f"http_{exc.status_code}",
        message=str(exc.detail),
    )
    # payload.model_dump()  将payload转换为字典（假设payload是Pydantic模型）
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    payload = error_response(
        code="validation_error",
        message="Request validation failed.",
        details=exc.errors(),
    )
    # HTTP状态码422（UnprocessableEntity，语义错误）
    # 422状态码的含义： 服务器理解请求，但请求参数语义错误（类型错误、缺少字段等）。
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload.model_dump())


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # path=记录请求路径，如 / api / users / 123
    # 这一行是唯一的"病历本"：记录完整堆栈供开发排查
    logger.exception("unhandled_exception", error=str(exc), path=request.url.path)
    payload = error_response(
        code="internal_server_error",
        message="Internal server error.",  # ← 用户只看到这个
    )
    # 返回给用户的只有通用信息（不暴露细节）
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump())
