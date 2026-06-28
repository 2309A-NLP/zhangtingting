"""问答相关的 Pydantic Schema"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequestSchema(BaseModel):
    """单轮问答请求"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    model: Optional[str] = Field(None, description="模型名称（可选）")
    temperature: Optional[float] = Field(None, ge=0.0, le=0.5, description="温度参数（SQL 生成建议 ≤0.2）")
    enable_few_shot: bool = Field(True, description="是否启用 Few-shot 检索")
    session_id: Optional[str] = Field("", description="会话 ID")


class ChatResponseSchema(BaseModel):
    """问答响应"""
    code: int = Field(0, description="状态码")
    message: str = Field("success", description="状态消息")
    data: dict[str, Any] = Field(default_factory=dict, description="响应数据")


class ChatErrorSchema(BaseModel):
    """错误响应"""
    code: int
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class BatchRequestSchema(BaseModel):
    """批量问答请求"""
    questions: list[str] = Field(..., min_length=1, max_length=1000, description="问题列表")


class BatchResponseSchema(BaseModel):
    """批量响应"""
    code: int
    message: str
    data: dict[str, Any]


class BatchStatusSchema(BaseModel):
    """批量状态"""
    code: int
    message: str
    data: dict[str, Any]


class TableInfoSchema(BaseModel):
    """表信息"""
    table_name: str
    description: str = ""
    columns: list[dict[str, str]] = Field(default_factory=list)
