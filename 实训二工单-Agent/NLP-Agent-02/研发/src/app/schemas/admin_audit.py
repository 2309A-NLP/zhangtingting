from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AdminAccessAuditLogCreate(BaseModel):
    path: str = Field(min_length=1, max_length=255)
    method: str = Field(min_length=1, max_length=16)
    client_host: str | None = Field(default=None, max_length=64)
    request_id: str | None = Field(default=None, max_length=64)
    access_granted: bool
    auth_mode: str = Field(min_length=1, max_length=32)
    failure_reason: str | None = Field(default=None, max_length=255)


class AdminAccessAuditLogRead(AdminAccessAuditLogCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


class AdminAccessAuditLogList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AdminAccessAuditLogRead]
