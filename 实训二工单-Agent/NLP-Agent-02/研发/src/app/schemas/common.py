from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int
    message: str
    data: T


class ErrorResponse(BaseModel):
    code: int
    message: str
    error_code: str
    request_id: str
    details: list[dict[str, Any]] | None = None
    data: None = None
