from __future__ import annotations
"""
统一API响应格式模块

成功响应格式：{success: true, request_id: "...", data: {...}}
错误响应格式：{success: false, request_id: "...", error: {code, message, details}}

包一层的必要性：
1.每个接口返回格式都不一样，前端要针对每个接口写不同的解析逻辑。==>统一格式，前端好处理
2.自动添加 request_id，方便链路追踪和问题排查
3.错误格式统一
4.方便以后扩展,统一新增的字段可以直接写在这里
包一层就像快递盒子——商品（业务数据）可以直接拿，但加上盒子后，可以统一贴快递单（request_id）、统一写地址、统一加缓冲材料。
"""

# Generic	泛型基类，用于定义支持泛型的类
# TypeVar	类型变量，用于定义泛型中的占位符类型
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from app.core.request_context import get_request_id

# 定义类型变量T，用于泛型类型注解
T = TypeVar("T")


class ErrorDetail(BaseModel):
    """
    错误详情模型

    Attributes:
        code: 错误码，用于程序识别错误类型
        message: 错误消息，用于展示给用户
        details: 错误详细信息，可选，用于调试
    """
    code: str
    message: str
    details: Any | None = None


class ApiResponse(BaseModel, Generic[T]):
    """
    成功响应模型

    Attributes:
        success: 是否成功，固定为True
        request_id: 请求ID，用于链路追踪
        data: 响应数据，泛型类型，可以是任意类型

    Type Parameters:
        T: 响应数据的类型
    """
    success: bool = True
    request_id: str | None = None
    data: T


class ApiErrorResponse(BaseModel):
    """
    错误响应模型

    Attributes:
        success: 是否成功，固定为False
        request_id: 请求ID，用于链路追踪
        error: 错误详情对象
    """
    success: bool = False
    request_id: str | None = None
    error: ErrorDetail


def success_response(data: T) -> ApiResponse[T]:
    """
    创建成功响应

    Args:
        data: 响应数据

    Returns:
        ApiResponse对象，包含请求ID和响应数据

    Example:
        >>> success_response({"user_id": "123"})
        ApiResponse(success=True, request_id="abc-123", data={"user_id": "123"})
    """
    # ApiResponse[T]这里的T是由(data: T)这里的T的类型自动推断出的
    # [T]指ApiResponse 这个类需要接收一个类型作为参数。
    return ApiResponse[T](request_id=get_request_id(), data=data)


def error_response(*, code: str, message: str, details: Any | None = None) -> ApiErrorResponse:
    """
    创建错误响应

    Args:
        code: 错误码
        message: 错误消息
        details: 错误详细信息，可选

    Returns:
        ApiErrorResponse对象，包含请求ID和错误详情

    Example:
        >>> error_response(code="USER_NOT_FOUND", message="用户不存在")
        ApiErrorResponse(success=False, request_id="abc-123", 
                      error=ErrorDetail(code="USER_NOT_FOUND", message="用户不存在"))
    """
    return ApiErrorResponse(
        request_id=get_request_id(),
        error=ErrorDetail(code=code, message=message, details=details),
    )
