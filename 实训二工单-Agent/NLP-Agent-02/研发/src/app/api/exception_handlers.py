# FastAPI 全局异常处理器，它的作用是统一处理所有异常，确保 API 返回的响应格式始终保持一致。
# 这段代码为 FastAPI 应用注册了 4 层全局异常处理器：资源未找到（404）、业务错误（4xx）、参数验证错误（422）、
# 未知错误（500）。所有错误响应格式统一为 {code, message, error_code, request_id, data, details}，
# 同时自动记录结构化日志，让排查问题变得简单。
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from structlog import get_logger

from app.core.exceptions import ApplicationError, NotFoundError
from app.schemas.common import ErrorResponse

logger = get_logger()
'''
exception_handlers.py
├── 辅助函数
│   ├── _get_request_id()      → 从请求中提取 request_id
│   └── _build_error_response() → 构造统一的错误响应格式
└── register_exception_handlers()
    ├── NotFoundError           → 404 资源未找到
    ├── ApplicationError        → 业务逻辑错误
    ├── RequestValidationError  → 请求参数验证失败（422）
    └── Exception               → 兜底捕获所有其他异常（500）
'''

def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")
'''
作用： 从请求对象中提取 request_id
工作原理：
每个请求进来时，中间件会生成一个唯一的 request_id
保存在 request.state.request_id 中
这里用 getattr 安全读取，如果没有就返回 "unknown"
用途： 在日志和响应中携带 request_id，方便追踪一次完整请求的调用链。
'''

# 作用： 构造统一的错误响应格式
def _build_error_response(
    *,
    status_code: int,
    message: str,
    error_code: str,
    request_id: str,
    details: list[dict[str, object]] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        code=status_code,
        message=message,
        error_code=error_code,
        request_id=request_id,
        details=details,
    )
    # model_dump() 是 Pydantic v2 的方法（v1 中为 dict()）
    # exclude_none=True 会过滤掉值为 None 的字段，避免响应中出现冗余的空字段
    content = payload.model_dump(exclude_none=True)
    content["data"] = None
    return JSONResponse(status_code=status_code, content=content)

# 注册异常处理器
def register_exception_handlers(app: FastAPI) -> None:
    # NotFoundError（404）
    # 使用 @app.exception_handler 装饰器注册特定异常的处理函数
    # 当应用中抛出 NotFoundError 时，自动触发此处理器
    # 返回统一的 JSONResponse 格式
    @app.exception_handler(NotFoundError)
    async def handle_not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        request_id = _get_request_id(request)
        logger.info(
            "request_not_found",
            request_id=request_id,
            path=request.url.path,
            message=exc.message,
        )
        return _build_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            request_id=request_id,
        )

    # ApplicationError（业务逻辑错误）
    @app.exception_handler(ApplicationError)
    async def handle_application_error(request: Request, exc: ApplicationError) -> JSONResponse:
        request_id = _get_request_id(request)
        logger.info(
            "request_application_error",
            request_id=request_id,
            path=request.url.path,
            error_code=exc.error_code,
            message=exc.message,
        )
        return _build_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            request_id=request_id,
        )

    # RequestValidationError（422）  请求体缺少必填字段\字段类型错误（传了字符串但期望数字）\字段值超出范围
    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = _get_request_id(request)
        logger.info(
            "request_validation_error",
            request_id=request_id,
            path=request.url.path,
            error_count=len(exc.errors()),
        )
        return _build_error_response(
            status_code=422,
            message="Request validation failed",
            error_code="REQUEST_VALIDATION_ERROR",
            request_id=request_id,
            details=jsonable_encoder(exc.errors()),
        )

    # 兜底处理器 Exception（500）
    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = _get_request_id(request)
        logger.exception(
            "request_unexpected_error",
            request_id=request_id,
            path=request.url.path,
        )
        return _build_error_response(
            status_code=500,
            message="Internal server error",
            error_code="INTERNAL_SERVER_ERROR",
            request_id=request_id,
        )
