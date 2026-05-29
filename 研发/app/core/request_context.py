from __future__ import annotations
'''
用户发起请求 → 中间件设置 request_id → 任何地方都能用，不需要在每个函数参数中传递。
request_id 什么时候生成？	请求到达服务器时
request_id 什么时候销毁？	请求处理完毕后
为什么这么设计？	方便追踪单个请求的完整链路

ContextVar比全局变量的好处是不会被覆盖，比局部变量的好处是不用处处传递，只要导入方法就行了
ContextVar = 全局变量的便利性 + 局部变量的隔离性
'''
# ContextVar 是什么？  ContextVar 是 Python 的"协程局部变量"——在同一请求链中全局可访问，在不同请求之间自动隔离，不会串数据
# 当前协程及子协程
# 在同一协程/任务中，无论调用多深，都能访问到同一个值
# 异步请求之间自动隔离，不会串数据
# 和"请求"或更准确地说，和"协程执行链"绑定。
# 父协程的上下文会自动复制给子协程
'''
ContextVar 和请求绑定（因为每个请求在独立的协程中处理），
通过 await 调用子协程时会自动复制上下文，
因此深层函数无需传参就能获取请求ID。
日志系统是最常用的场景，让你在所有日志中自动包含 request_id，方便追踪整个请求链路。

谁用 await 调用别人，谁就是父协程（调用者）；被 await 调用的就是子协程（被调用者）。
'''
from contextvars import ContextVar

# 创建上下文变量
_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
# : ContextVar[str | None]	类型注解：这是一个存储 str 或 None 的上下文变量
# ContextVar("request_id", default=None)	创建一个名为 "request_id" 的上下文变量，默认值为 None


def set_request_id(request_id: str) -> None:
    _request_id_ctx.set(request_id)


def get_request_id() -> str | None:
    return _request_id_ctx.get()
