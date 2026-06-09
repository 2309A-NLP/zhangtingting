from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class CitationSummary(BaseModel):
    chunk_id: str = ""
    page_number: int = 0
    doc_name: str = ""
    profile_id: str = ""


class ConversationTurn(BaseModel):
    query: str
    rewritten_query: str = ""
    answer_summary: str = ""
    resolved_company: str = ""
    resolved_profile_id: str = ""
    question_type: str = ""
    current_subject: str = ""
    used_history: bool = False
    rewrite_reason: str = ""
    citations: list[CitationSummary] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationState(BaseModel):
    session_id: str
    current_company: str = ""
    current_profile_id: str = ""
    current_topic: str = ""
    current_question_type: str = ""
    current_subject: str = ""
    last_rewritten_query: str = ""
    last_answer_summary: str = ""
    last_citations: list[CitationSummary] = Field(default_factory=list)
    history_turns: list[ConversationTurn] = Field(default_factory=list)
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResolvedQuery(BaseModel):
    original_query: str
    rewritten_query: str
    resolved_company: str = ""
    resolved_profile_id: str = ""
    question_type: str = ""
    current_subject: str = ""
    used_history: bool = False
    rewrite_reason: str = ""
    resolution_mode: Literal["passthrough", "history_rewrite"] = "passthrough"
