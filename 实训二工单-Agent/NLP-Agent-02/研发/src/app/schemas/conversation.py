from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConversationStateCreate(BaseModel):
    session_id: str
    intent: str
    agent_state: str
    tool_name: str | None = None
    tool_arguments_json: str | None = None
    user_message: str | None = None
    expires_at: datetime | None = None


class ConversationStateRead(ConversationStateCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


class ConversationSessionView(BaseModel):
    id: int
    session_id: str
    intent: str
    agent_state: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    user_message: str | None = None
    expires_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    is_expired: bool
    has_pending_confirmation: bool


class ConversationSessionList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ConversationSessionView]
