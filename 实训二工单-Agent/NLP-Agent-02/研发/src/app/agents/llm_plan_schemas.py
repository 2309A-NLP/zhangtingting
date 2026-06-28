from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMPlanResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    complexity: str
    intent: str
    action: str
    extracted: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    reply_style: str
    reasoning_summary: str

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, value: str) -> str:
        if value not in {"simple", "complex"}:
            raise ValueError(f"Unsupported complexity: {value}")
        return value

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, value: str) -> str:
        if value not in {"create", "query", "update", "delete", "unknown"}:
            raise ValueError(f"Unsupported intent: {value}")
        return value

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        if value not in {"confirm", "clarify", "execute", "reply"}:
            raise ValueError(f"Unsupported action: {value}")
        return value
