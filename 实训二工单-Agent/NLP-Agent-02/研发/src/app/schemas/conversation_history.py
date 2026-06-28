import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConversationHistoryCreate(BaseModel):
    session_id: str
    user_input: str
    confirmed: bool
    context_json: str | None = None
    parser_source: str | None = None
    intent: str
    agent_state: str
    tool_name: str | None = None
    tool_arguments_json: str | None = None
    missing_fields_json: str | None = None
    suggested_inputs_json: str | None = None
    target_id: int | None = None
    execution_result_json: str | None = None
    user_message: str | None = None


class ConversationHistoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
    user_input: str
    confirmed: bool
    parser_source: str | None = None
    intent: str
    agent_state: str
    tool_name: str | None = None
    target_id: int | None = None
    user_message: str | None = None
    created_at: datetime
    updated_at: datetime
    context: dict[str, Any] = Field(default_factory=dict)
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    suggested_inputs: list[str] = Field(default_factory=list)
    execution_result: Any = None
    context_json: str | None = Field(default=None, exclude=True)
    tool_arguments_json: str | None = Field(default=None, exclude=True)
    missing_fields_json: str | None = Field(default=None, exclude=True)
    suggested_inputs_json: str | None = Field(default=None, exclude=True)
    execution_result_json: str | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def transform_from_source(cls, value: Any) -> Any:
        if hasattr(value, "__dict__"):
            value = {key: getattr(value, key) for key in value.__dict__ if not key.startswith("_")}
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data["context"] = cls._load_dict(data.get("context_json"))
        data["tool_arguments"] = cls._load_dict(data.get("tool_arguments_json"))
        data["missing_fields"] = cls._load_list(data.get("missing_fields_json"))
        data["suggested_inputs"] = cls._load_list(data.get("suggested_inputs_json"))
        data["execution_result"] = cls._load_any(data.get("execution_result_json"))
        return data

    @model_validator(mode="after")
    def populate_decoded_payloads(self) -> "ConversationHistoryRead":
        self.context = self._load_dict(self.context_json)
        self.tool_arguments = self._load_dict(self.tool_arguments_json)
        self.missing_fields = self._load_list(self.missing_fields_json)
        self.suggested_inputs = self._load_list(self.suggested_inputs_json)
        self.execution_result = self._load_any(self.execution_result_json)
        return self

    @staticmethod
    def _load_dict(value: str | None) -> dict[str, Any]:
        loaded = ConversationHistoryRead._load_any(value)
        if isinstance(loaded, dict):
            return loaded
        return {}

    @staticmethod
    def _load_list(value: str | None) -> list[str]:
        loaded = ConversationHistoryRead._load_any(value)
        if isinstance(loaded, list):
            return [str(item) for item in loaded]
        return []

    @staticmethod
    def _load_any(value: str | None) -> Any:
        if value is None:
            return None
        return json.loads(value)


class ConversationHistoryQuery(BaseModel):
    session_id: str | None = None
    parser_source: str | None = None
    agent_state: str | None = None
    intent: str | None = None
    limit: int = 20
    offset: int = 0


class ConversationHistoryList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ConversationHistoryRead]
