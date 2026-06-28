"""通用装饰器 — 性能监控、日志、重试"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


def timing(func: F) -> F:
    """计时装饰器 — 记录函数执行时间"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} 执行耗时: {elapsed:.2f}ms")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """重试装饰器 — 失败时自动重试"""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} 第 {attempt + 1} 次尝试失败: {e}, "
                            f"{current_delay:.1f}s 后重试..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} 重试 {max_attempts} 次后仍失败")
            raise last_exception
        return wrapper
    return decorator


def log_call(level: str = "INFO"):
    """日志装饰器 — 记录函数调用"""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = getattr(logger, level.lower())
            log(f"调用 {func.__name__}(args={args[:2] if args else ()}, kwargs={list(kwargs.keys())})")
            try:
                result = func(*args, **kwargs)
                log(f"{func.__name__} 执行成功")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} 执行失败: {e}")
                raise
        return wrapper
    return decorator


def cache_result(ttl: int = 300):
    """简单结果缓存装饰器"""
    def decorator(func: F) -> F:
        cache: dict[str, tuple[float, Any]] = {}
        lock = __import__("threading").Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            with lock:
                if key in cache:
                    timestamp, result = cache[key]
                    if time.time() - timestamp < ttl:
                        logger.debug(f"缓存命中: {key}")
                        return result
            result = func(*args, **kwargs)
            with lock:
                cache[key] = (time.time(), result)
            return result
        wrapper.cache = cache
        return wrapper
    return decorator
