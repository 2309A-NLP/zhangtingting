from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

VALID_AGENT_STATES = {"confirm", "clarify", "execute", "reply"}
VALID_INTENTS = {"create", "query", "update", "delete", "unknown"}
VALID_TOOL_NAMES = {
    "schedule_create",
    "schedule_list",
    "schedule_get",
    "schedule_update",
    "schedule_delete",
    "conversation_cancel",
}


class LLMParseResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent_state: str
    intent: str
    user_message: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    target_id: int | None = None

    @field_validator("agent_state")  # 装饰器：指定要校验的字段
    @classmethod
    def validate_agent_state(cls, value: str) -> str:
        if value not in VALID_AGENT_STATES:
            raise ValueError(f"Unsupported agent_state: {value}")
        return value

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, value: str) -> str:
        if value not in VALID_INTENTS:
            raise ValueError(f"Unsupported intent: {value}")
        return value

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str | None) -> str | None:
        if value is not None and value not in VALID_TOOL_NAMES:
            raise ValueError(f"Unsupported tool_name: {value}")
        return value
