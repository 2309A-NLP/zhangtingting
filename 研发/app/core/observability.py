from __future__ import annotations
'''
运行时追踪（Runtime Trace）装饰器模块
📦 模块整体功能
这个模块提供了：
内容预览 - 截断过长的日志内容
负载归一化 - 将复杂数据结构转为可记录格式
运行时追踪 - 发送结构化追踪日志
计时装饰器 - 自动记录函数执行时间
'''
# 内省模块，用于检查函数类型  用途：判断函数是异步还是同步函数
import inspect
import time
# 可调用对象类型
# 用途：类型注解，表示装饰器返回的函数类型
from collections.abc import Callable
# 装饰器工具
# 用途：保留被装饰函数的元数据（name, __doc__等）
from functools import wraps
from typing import Any, TypeVar

from app.core.config import get_settings

# 作用：定义类型变量 F
# bound=Callable[..., Any]：F 必须是可调用对象
# 用于保持装饰器输入和输出的类型一致
F = TypeVar("F", bound=Callable[..., Any])


def preview_text(value: Any, limit: int | None = None) -> str:
    settings = get_settings()
    # 获取预览长度限制
    preview_limit = limit or settings.app_runtime_content_preview_chars
    # 转换字符串并转义换行符  这样日志中不会有多行，保持单行格式
    text = str(value).replace("\n", "\\n")
    # 截断过长文本
    if len(text) <= preview_limit:
        return text
    return f"{text[:preview_limit]}...(truncated)"

# 负载归一化
def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    '''
    字符串类型 → 应用预览截断
    基础类型 → 直接保留（这些类型本身就很短）
    列表 → 只记录长度
    字典 → 只记录前8个键名
    其他类型 → 转为字符串并预览
    '''
    for key, value in payload.items():
        if isinstance(value, str):
            normalized[key] = preview_text(value)
        elif isinstance(value, (int, float, bool)) or value is None:
            normalized[key] = value
        elif isinstance(value, list):
            normalized[key] = f"list(len={len(value)})"
        elif isinstance(value, dict):
            normalized[key] = f"dict(keys={list(value.keys())[:8]})"
        else:
            normalized[key] = preview_text(value)
    return normalized

# 发送追踪日志
def emit_runtime_trace(logger: Any, event: str, **payload: Any) -> None:
    settings = get_settings()
    if not settings.app_runtime_trace:
        return

    normalized = _normalize_payload(payload)
    logger.info(event, **normalized)

    if settings.app_runtime_print:
        # 作用：可选的控制台打印（开发调试用）
        # flush=True：立即刷新输出，不缓冲
        print(f"[TRACE] {event} | {normalized}", flush=True)

# 核心计时装饰器

# 作用：装饰器工厂（带参数的装饰器）
# event_name：事件名称（如 "chat_completion"）
# emit_start：是否记录开始事件
# 返回真正的装饰器函数
def log_timed(event_name: str, *, emit_start: bool = True) -> Callable[[F], F]:
    # 判断函数类型  使用 inspect 判断被装饰函数是同步还是异步
    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):
            # 异步函数处理
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                logger = getattr(args[0], "logger", None) if args else None
                started = time.perf_counter()
                if logger is not None and emit_start:
                    emit_runtime_trace(logger, f"{event_name}_started")
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    if logger is not None:
                        emit_runtime_trace(
                            logger,
                            f"{event_name}_failed",
                            elapsed_ms=elapsed_ms,
                            error=str(exc),
                        )
                    raise

                elapsed_ms = int((time.perf_counter() - started) * 1000)
                if logger is not None:
                    emit_runtime_trace(logger, f"{event_name}_finished", elapsed_ms=elapsed_ms)
                return result

            return async_wrapper  # type: ignore[return-value]

        # 同步函数处理
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = getattr(args[0], "logger", None) if args else None
            started = time.perf_counter()
            if logger is not None and emit_start:
                emit_runtime_trace(logger, f"{event_name}_started")
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                if logger is not None:
                    emit_runtime_trace(
                        logger,
                        f"{event_name}_failed",
                        elapsed_ms=elapsed_ms,
                        error=str(exc),
                    )
                raise

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            if logger is not None:
                emit_runtime_trace(logger, f"{event_name}_finished", elapsed_ms=elapsed_ms)
            return result

        return sync_wrapper  # type: ignore[return-value]

    return decorator
