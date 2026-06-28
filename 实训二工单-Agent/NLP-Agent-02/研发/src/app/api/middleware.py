'''
这个中间件在每个请求处理前后执行，完成三件事：
生成/提取请求ID（用于追踪）
绑定日志上下文（让所有日志带上请求信息）
在响应头中返回请求ID（方便客户端关联）
'''
from collections.abc import Awaitable, Callable
# Awaitable 是指可以被 await 关键字等待的对象，即协程。
# Callable 是指可以被调用（使用 () 语法）的对象，如函数、方法、类、实现了 __call__ 的对象。
from uuid import uuid4

from fastapi import Request, Response
from structlog.contextvars import bind_contextvars, clear_contextvars


async def request_context_middleware(
    request: Request,
    # 描述 call_next 是一个接收 Request 返回 Awaitable[Response] 的异步函数。
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    '''
    优先使用请求头中的 X-Request-ID：
    如果客户端在请求头中传了 X-Request-ID（如从前端、网关或上游服务传递），则使用该值。
    这是分布式追踪的标准做法，让链路追踪 ID 保持一致。
    否则生成新 ID：
    使用 uuid4().hex 生成 32 位十六进制随机字符串（如 "f47ac10b58cc4372a5670e02b2c3d479"）。
    存储到 request.state：
    request.state 是 FastAPI 提供的请求级别存储空间，可以在整个请求生命周期中访问。
    后续的路由、Service、工具函数都可以通过 request.state.request_id 获取该 ID。
    '''

    clear_contextvars()
    bind_contextvars(request_id=request_id, path=request.url.path, method=request.method)
    '''
    clear_contextvars()：
    清空当前协程的上下文变量。
    为什么需要？ 因为 FastAPI 使用异步协程处理请求，协程可能被复用（如线程池），之前的日志上下文可能残留在 contextvars 中，导致日志串扰。
    bind_contextvars(...)：
    将 request_id、path、method 绑定到日志上下文。
    之后所有使用 structlog 记录的日志都会自动携带这些字段。
    '''

    try:
        response = await call_next(request)
    finally:
        clear_contextvars()
    '''
    await call_next(request)：
    调用下一个中间件或最终的路由处理器。
    这是实际执行业务逻辑的地方。
    try...finally 保证清理：
    无论请求成功还是抛出异常，都会执行 finally 块。
    再次 clear_contextvars() 确保请求结束后上下文被清空，避免污染下一个请求。
    '''

    '''
    将 request_id 添加到响应头中。
    作用：
    客户端可关联日志：如果请求出错，客户端可以将 X-Request-ID 提供给技术支持，服务端根据该 ID 快速定位日志。
    链路追踪：下游服务可以读取该 ID 并继续传递，实现跨服务追踪。
    '''
    response.headers["X-Request-ID"] = request_id
    return response

'''
客户端发起请求（可能带 X-Request-ID）
    ↓
中间件接收请求
    ↓
1. 提取/生成 request_id
   存入 request.state.request_id
    ↓
2. 清空旧上下文（clear_contextvars）
   绑定新上下文（bind_contextvars）
    ↓
3. 调用 next 中间件/路由
    ├── 路由处理
    ├── Service 层调用
    ├── 日志记录（自动携带 request_id）
    ├── 异常处理
    └── 返回响应
    ↓
4. finally 清空上下文（clear_contextvars）
    ↓
5. 响应头添加 X-Request-ID
    ↓
返回响应给客户端
'''