from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentNaturalLanguageRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    user_input: str = Field(min_length=1)
    confirmed: bool = False
    context: dict[str, Any] = Field(default_factory=dict)


class AgentToolRequest(BaseModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentToolResult(BaseModel):
    success: bool
    tool_name: str
    data: Any = None
    error: str | None = None


class AgentExecuteResponse(BaseModel):
    success: bool
    system_prompt: str
    tool_result: AgentToolResult


class AgentStateResponse(BaseModel):
    agent_state: str
    intent: str
    user_message: str
    parser_source: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    suggested_inputs: list[str] = Field(default_factory=list)
    target_id: int | None = None
    execution_result: Any = None
