import logging
import sys
from typing import Any

import structlog

from app.core.config import get_settings


def _shared_processors() -> list[Any]:
    return [
        # 合并上下文变量  在异步编程中（如 FastAPI），每个请求可能在不同的上下文/协程中运行。  输出：{"event": "查询数据库", "request_id": "req-12345", ...}
        structlog.contextvars.merge_contextvars,
        # 添加日志器名称  当天日志量很大时，可以通过 logger 字段快速定位是哪段代码输出的日志 输出：{"event": "用户登录", "logger": "myapp.user.service", ...}
        structlog.stdlib.add_logger_name,
        # 添加日志级别 添加level字段  {"event": "错误消息", "level": "error", ...}
        structlog.stdlib.add_log_level,
        # 添加时间戳  在每条日志中添加一个 timestamp 字段
        # fmt 值	             输出示例	                             说明
        # "iso"	            "2024-01-15T10:30:45.123456"	    ISO 8601 标准格式
        # "iso" 带 utc=True	"2024-01-15T10:30:45.123456Z"	    末尾加 Z 表示 UTC 时间
        # "iso" 带 utc=False	"2024-01-15T10:30:45.123456+08:00"	本地时区偏移
        # 其他字符串	          按字符串格式输出	                     较少使用
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        # 堆栈信息渲染器  当你在日志中调用 logger.info("xxx", stack_info=True) 时，这个处理器会收集并格式化堆栈信息。
        structlog.processors.StackInfoRenderer(),
        # 格式化异常信息   当你在 except 块中调用 logger.error("...", exc_info=True) 或 logger.exception() 时，它会捕获异常信息并格式化。
        structlog.processors.format_exc_info,
    ]

# 这个函数应该在应用启动时调用一次（比如在 main.py 或 __init__.py 中）
def setup_logging() -> None:
    settings = get_settings()  # 获取应用配置，通常来自环境变量或 .env 文件。

    # basicConfig()  logging模块的一个函数，用于基础一次性配置日志系统
    # 配置日志系统的全局设置。这个函数应该在程序启动时调用一次。
    '''
    为什么只输出 %(message)s？
        因为 structlog 会负责添加时间戳、日志级别、JSON 格式化等，标准 logging 只需要输出最终的 JSON 字符串。
    为什么用 sys.stdout？
        在 Docker/Kubernetes 环境中，容器运行时只收集 stdout 和 stderr
        统一输出到一个流更容易收集和处理
        structlog 的 JSON 输出不是错误信息，所以用 stdout 更合适
    '''
    logging.basicConfig(
        format="%(message)s",   # 指定日志输出的格式
                                # % (message)s 中的message是一个占位符，代表日志记录中的"消息"部分
                                # % 表示这是格式化占位符
                                # (message) 指定要使用的字段名
                                # s 表示以字符串格式输出
        stream=sys.stdout,  # 指定日志输出的目标流   stdout 标准输出流（通常是控制台/终端）   sys.stderr 标准错误
        # level 作用： 设置全局日志级别，只有高于或等于这个级别的日志才会被输出。
        level=getattr(logging, settings.app_log_level.upper(), logging.INFO),  # 配置日志级别
        # settings.app_log_level.upper()  将配置中的日志级别字符串转成大写，确保后面能正确匹配
        # getattr 的作用：从对象中获取属性
        # 相当于：
        # 尝试 logging.INFO
        # 如果 logging 模块中有 "INFO" 这个属性，就返回它
        # 如果没有，返回默认值 logging.INFO
        # getattr(对象, "属性名", 默认值)
        # 返回：对象.属性名  如果属性不存在，返回默认值
    )
    '''
    日志级别对照表：
    级别名	   整数值
    DEBUG	    10
    INFO	    20
    WARNING	    30
    ERROR	    40
    CRITICAL	50
    '''

    # structlog 第三方库，用于结构化日志
    # configure 配置函数，用于设置结构化日志的行为
    structlog.configure(
        # 指定一个处理器列表
        processors=[
            # 处理器是一个函数，接收日志事件（一个字典），处理后返回修改后的字典。
            # 展开共享处理器
            *_shared_processors(),  # *星号展开运算符
            # structlog.processors    structlog 内置的处理器模块
            # JSONRenderer()    创建一个 JSON 渲染器处理器实例 将日志字典转换成 JSON 字符串
            # 这是处理器链的最后一环：所有前面的处理器添加完字段后，最后转换成JSON
            structlog.processors.JSONRenderer(),
        ],
        # 指定日志器的包装类
        wrapper_class=structlog.make_filtering_bound_logger(  # structlog 的一个工厂函数，创建带过滤功能的日志器包装类
            getattr(logging, settings.app_log_level.upper(), logging.INFO),
        ),
        # 指定创建底层日志器的工厂
        logger_factory=structlog.stdlib.LoggerFactory(),  # 创建一个工厂实例，它会使用 Python 标准库的 logging.getLogger()
        cache_logger_on_first_use=True,  # 开启缓存
    )
    '''
    为什么需要这个？logger_factory=
        structlog 只是"包装器"，最终日志还是要交给标准库 logging 处理
        LoggerFactory() 创建的日志器会将日志事件发送到 logging 模块
        然后 logging 模块根据 basicConfig 的配置输出
    '''

# 返回一个 structlog 绑定的日志器对象
def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    # 调用structlog获取一个日志器实例。
    return structlog.get_logger(name or get_settings().app_name)
