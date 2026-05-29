from __future__ import annotations
''''
定义了整个 API 层所有请求和响应的数据结构（Pydantic 模型）。
前后端之间的"数据契约"（接口协议）
'''
from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

# 聊天请求体
class ChatRequest(BaseModel):
    user_id: str
    role_id: str | None = None
    role_name: str | None = None
    query: str = Field(min_length=1)
    stream: bool = False
    session_id: str | None = None
    top_k: int = Field(default=8, ge=0, le=20)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_role_selector(self) -> "ChatRequest":
        if not self.role_id and not self.role_name:
            raise ValueError("Either role_id or role_name must be provided.")
        return self

# 引用来源(前端用) == chat/models.py
class ContextSourceSchema(BaseModel):
    doc_id: str
    chunk_id: str
    source: str
    score: float

# 聊天响应
class ChatResponse(BaseModel):
    request_id: str
    role_id: str
    role_name: str
    session_id: str
    response: str
    context_sources: list[ContextSourceSchema]
    tokens_used: int
    latency_ms: int
    model: str
    degraded_to_online_api: bool
    rewritten_query: str

# 清除对话历史记忆请求
class ClearChatRequest(BaseModel):
    user_id: str
    role_id: str
    session_id: str | None = None

# 清除对话历史记忆响应
class ClearChatResponse(BaseModel):
    success: bool
    cleared_keys: list[str]
    session_id: str | None = None

# 角色数据结构
class RoleSchema(BaseModel):
    role_id: str
    name: str
    category: str
    # preset	系统预置角色（不可修改）
    # custom	用户自定义角色
    # auto	自动检测角色
    role_type: Literal["preset", "custom", "auto"]
    system_prompt: str
    knowledge_base_id: str | None = None
    created_at: datetime | None = None

# 角色列表响应。
class RolesListResponse(BaseModel):
    total: int
    items: list[RoleSchema]

# 角色匹配请求
class RoleDetectRequest(BaseModel):
    user_id: str
    query: str = Field(min_length=1)

#  角色匹配响应
class RoleDetectResponse(BaseModel):
    role_id: str
    role_name: str
    category: str
    # matched	匹配到已有角色
    # created	自动创建了新角色
    # assigned	被分配了角色
    action: Literal["matched", "created", "assigned"]
    confidence: float
    reason: str

# 删除自定义角色响应
class DeleteCustomRoleResponse(BaseModel):
    success: bool
    role_id: str

# 创建自定义角色响应
class CreateCustomRoleRequest(BaseModel):
    user_id: str
    name: str = Field(min_length=1, max_length=64)
    system_prompt: str = Field(min_length=1)
    category: str = "general"
    knowledge_base_id: str | None = None

# 更新自定义角色响应
class UpdateCustomRoleRequest(BaseModel):
    user_id: str
    name: str = Field(min_length=1, max_length=64)
    system_prompt: str = Field(min_length=1)
    category: str = "general"
    knowledge_base_id: str | None = None

# 知识库上传响应
class KnowledgeUploadResponse(BaseModel):
    task_id: str
    user_id: str
    role_id: str
    mode: Literal["full", "incremental"]
    status: Literal["queued", "processing", "success", "failed"]
    overwrite: bool = False
    duplicate_of_file_id: str | None = None
    uploaded_at: datetime

# 知识库任务状态响应
# 轮询此接口获取处理进度
class KnowledgeTaskStatusResponse(BaseModel):
    task_id: str
    user_id: str
    role_id: str
    mode: Literal["full", "incremental"]
    status: Literal["queued", "processing", "success", "failed"]
    doc_id: str | None = None
    source_uri: str | None = None
    parsed_artifact_uri: str | None = None
    chunk_count: int | None = None
    error_message: str | None = None
    started_at: int | None = None
    finished_at: int | None = None

# 健康检查
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    services: dict[str, str]

# 注册请求 用户名 3-64 字符，密码 8-128 字符。
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)
    email: str | None = Field(default=None, max_length=128)

# 登陆请求
class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)

# 登录成功后返回
class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in_seconds: int
    user_id: str
    username: str

# 用户信息响应   /auth/me 接口返回。
class UserProfileResponse(BaseModel):
    user_id: str
    username: str
    email: str | None = None

# 定义一个类型变量 T，代表"某种类型"，具体是什么类型由使用时决定。
# "T" 就是一个名字/标签，用来标识这个类型变量，调试时会显示这个名字
T = TypeVar("T")

# 继承 Pydantic 的 BaseModel（数据验证），同时继承 Generic[T]（泛型支持）
# 使用泛型支持 是为了后面写代码有方法提醒
# Generic[T] 是对变量 T 使用泛型支持
class Envelope(BaseModel, Generic[T]):
    # 固定字段，表示操作是否成功（默认 True）。
    success: bool = True
    # 请求追踪 ID，用于链路追踪。
    request_id: str | None = None
    # 实际响应数据，类型由 T 决定（灵活）。
    data: T
