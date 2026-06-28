import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class LLMAuditLogCreate(BaseModel):
    session_id: str
    user_input: str
    parser_stage: str
    provider: str | None = None
    model_name: str | None = None
    success: bool
    request_payload_json: str | None = None
    raw_response_text: str | None = None
    parsed_response_json: str | None = None
    error_message: str | None = None


class LLMAuditLogRead(LLMAuditLogCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime
    request_payload: dict[str, Any] | None = None
    parsed_response: dict[str, Any] | None = None

    @model_validator(mode="after")
    def populate_decoded_payloads(self) -> "LLMAuditLogRead":
        self.request_payload = self._load_json(self.request_payload_json)
        self.parsed_response = self._load_json(self.parsed_response_json)
        return self

    @staticmethod
    def _load_json(value: str | None) -> dict[str, Any] | None:
        if value is None:
            return None
        loaded = json.loads(value)
        if not isinstance(loaded, dict):
            return None
        return loaded


class LLMAuditLogQuery(BaseModel):
    session_id: str | None = None
    parser_stage: str | None = None
    success: bool | None = None
    limit: int = 50
    offset: int = 0


class LLMAuditLogList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[LLMAuditLogRead]
