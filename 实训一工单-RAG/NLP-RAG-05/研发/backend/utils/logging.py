# ??????????????? NLP-RAG-???????????????????

"""
??????????

??????
    from backend.utils.logging import get_logger, logged, log_stage

    logger = get_logger(__name__)

    @logged
    def my_function(arg1, arg2):
        ...

    with log_stage("PDF?????"):
        ...

?????Z??
    - logger.info()  ?? ???????????????C?/????????h
    - logger.warning() ?? ???????????? VLM API ???????????
    - logger.error()   ?? ?????????????????????
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

_LOGGER_initialized = False
P = ParamSpec("P")
R = TypeVar("R")


def _init_logging() -> None:
    global _LOGGER_initialized
    if _LOGGER_initialized:
        return

    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "project.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("nlp_rag")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    _LOGGER_initialized = True


def get_logger(name: str) -> logging.Logger:
    """???????????? logger??

    Args:
        name: ????? `__name__`??????????? 'nlp_rag.' ????

    Returns:
        ??????? logger ?????
    """
    _init_logging()
    if not name.startswith("nlp_rag."):
        name = f"nlp_rag.{name}"
    return logging.getLogger(name)


class log_stage:
    """?????????????????????????????????????????????????

    ??????
        with log_stage("PDF???????"):
            ...
    """

    def __init__(self, stage_name: str, logger: logging.Logger | None = None) -> None:
        self.stage_name = stage_name
        self.logger = logger or get_logger("pipeline.stage")
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "log_stage":
        self.start_time = time.perf_counter()
        self.logger.info("[???] %s", self.stage_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.elapsed = time.perf_counter() - self.start_time
        if exc_type is not None:
            self.logger.error(
                "[???] %s (%.2fs) | %s: %s",
                self.stage_name,
                self.elapsed,
                exc_type.__name__,
                exc_val,
            )
            self.logger.debug("Traceback:\n%s", "".join(traceback.format_exception(exc_type, exc_val, exc_tb)))
            return False  # do NOT suppress exceptions
        self.logger.info("[???] %s (%.2fs)", self.stage_name, self.elapsed)
        return False


def logged(func: Callable[P, R]) -> Callable[P, R]:
    """??????????????????????????

    ??????????????????????????????????????????????????????? Path???????????

    ???????????????????? Path ????????????????????

    ??????
        @logged
        def process_pdf(path: Path) -> list[Page]:
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger = get_logger(func.__module__)
        arg_summary = _summarize_args(func, args, kwargs)
        logger.info(">>> %s(%s)", func.__qualname__, arg_summary)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            output_summary = _summarize_output(result)
            logger.info("<<< %s (%.3fs) -> %s", func.__qualname__, elapsed, output_summary)
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "!!! %s (%.3fs) -> [%s] %s",
                func.__qualname__,
                elapsed,
                exc.__class__.__name__,
                exc,
            )
            logger.debug("Traceback:\n%s", "".join(traceback.format_exception(*sys.exc_info())))
            raise  # re-raise the original exception

    return wrapper


def _summarize_args(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """????????????????????"""
    try:
        import inspect

        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        parts: list[str] = []
        for i, arg in enumerate(args):
            name = param_names[i] if i < len(param_names) else f"arg{i}"
            parts.append(f"{name}={_safe_repr(arg)}")
        for k, v in kwargs.items():
            parts.append(f"{k}={_safe_repr(v)}")
        result = ", ".join(parts)
        return result if len(result) <= 200 else result[:197] + "..."
    except Exception:
        return f"args={_safe_repr(args)}, kwargs={_safe_repr(kwargs)}"


def _summarize_output(result: Any) -> str:
    """??????????????????????"""
    if result is None:
        return "None"
    if isinstance(result, Path):
        return f"Path({result.name})"
    if isinstance(result, (int, float, bool, str)):
        return repr(result)[:80]
    if isinstance(result, (list, tuple)):
        return f"{type(result).__name__}[len={len(result)}]"
    if isinstance(result, dict):
        return f"dict[keys={list(result.keys())[:5]}]"
    return f"{type(result).__name__}(id={id(result)})"


def _safe_repr(value: Any) -> str:
    """????????????????????????"""
    if value is None:
        return "None"
    if isinstance(value, Path):
        return f"Path({value.name})"
    if isinstance(value, (int, float, bool, str)):
        text = repr(value)
        return text[:60] + "..." if len(text) > 60 else text
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"{type(value).__name__}[len={len(value)}]"
    if isinstance(value, dict):
        keys = list(value.keys())[:5]
        return f"dict[keys={keys}]"
    return f"{type(value).__name__}(id={id(value)})"
