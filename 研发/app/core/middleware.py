from __future__ import annotations

import time
import uuid
import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from app.core.logging import get_logger
from app.core.request_context import set_request_id
logger = get_logger(__name__)
'''
进去时：往日志上下文加 request_id、path、method
出来时：往响应头加 X-Request-Id
'''

# 继承自 FastAPI/Starlette 的基类，提供异步 HTTP 中间件的基础功能
# 为什么继承 BaseHTTPMiddleware？
# 这是 FastAPI/Starlette 提供的便捷基类
# 你只需要实现 dispatch 方法即可
class RequestContextMiddleware(BaseHTTPMiddleware):
    # request：当前HTTP请求，包含路径、方法、headers、body等
    # call_next：下一个处理函数，调用它才能继续处理请求
    async def dispatch(self, request: Request, call_next):
        # request.headers.get("X-Request-Id") 从HTTP请求头中获取X - Request - Id字段（如果客户端传递了的话）
        # uuid.uuid4().hex  生成一个随机UUID，.hex将其转换为32位十六进制字符串（不含横杠）
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        # request.state  FastAPI / Starlette提供的请求级别存储区域，可以在中间件、路由函数之间传递数据
        # .request_id = request_id将请求ID存储到state中
        request.state.request_id = request_id
        # 自定义函数  设置一个上下文变量或线程局部变量，让非 HTTP 层的代码也能获取当前请求ID。
        set_request_id(request_id)
        # structlog.contextvars  structlog的上下文变量模块
        # clear_contextvars()清除当前协程 / 任务中绑定的所有上下文变量
        # FastAPI使用协程处理请求，协程可能被复用
        # 如果不清除，上一个请求的上下文变量会残留到下一个请求导致日志中出现错误的request_id、path等信息
        structlog.contextvars.clear_contextvars()
        # 绑定请求上下文
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )
        # 记录开始时间   time.perf_counter()	Python 高精度计时器，返回秒数（包含小数）
        # 为什么用perf_counter而不是time.time()？
        # perf_counter精度更高（纳秒级）
        # 不受系统时钟调整的影响（如NTP时间同步）
        started_at = time.perf_counter()
        try:
            # 会沿着中间件链继续向下，最终执行到你的路由函数（如 @ app.get("/hello")）。
            response = await call_next(request)
        finally:
            # *1000 变毫秒   int  去小数位
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            # locals()  # 获取当前作用域的所有局部变量（字典）
            # status_code = ...,  # HTTP 状态码
            # latency_ms = latency_ms  # 请求耗时（毫秒）
            logger.info("request_completed", status_code=getattr(locals().get("response"), "status_code", 500), latency_ms=latency_ms)
            # 清除为这个请求绑定的上下文变量，防止污染下一个复用的协程。
            structlog.contextvars.clear_contextvars()
        # 添加响应头
        # 客户端可以通过响应头知道这次请求的ID，便于问题排查时关联日志。
        response.headers["X-Request-Id"] = request_id
        return response
