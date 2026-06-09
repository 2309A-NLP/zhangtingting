from __future__ import annotations

from datetime import datetime, timezone

from backend.conversation.models import CitationSummary, ConversationState, ConversationTurn
from backend.conversation.query_understanding import resolve_query
from backend.conversation.store import ConversationStore
from backend.retrieval.unified_query_service import UnifiedDefaultQueryService
from backend.schemas import QueryResponse


class ConversationQueryService:
    def __init__(self, query_service: UnifiedDefaultQueryService, store: ConversationStore, history_limit: int = 6) -> None:
        self.query_service = query_service
        self.store = store
        self.history_limit = max(1, history_limit)

    def _build_answer_summary(self, response: QueryResponse) -> str:
        answer = str(response.answer or "").strip()
        if not answer:
            return ""
        compact = answer.replace("\n", " ").strip()
        return compact[:180]

    def _build_citation_summaries(self, response: QueryResponse) -> list[CitationSummary]:
        rows: list[CitationSummary] = []
        for item in response.citations[:5]:
            metadata = dict(item.metadata or {})
            rows.append(
                CitationSummary(
                    chunk_id=str(item.chunk_id or ""),
                    page_number=int(item.page_number or 0),
                    doc_name=str(metadata.get("doc_name") or ""),
                    profile_id=str(metadata.get("profile") or ""),
                )
            )
        return rows

    def _ensure_state(self, session_id: str, state: ConversationState | None) -> ConversationState:
        if state is not None:
            return state
        return ConversationState(session_id=session_id)

    def ask(self, query: str, session_id: str | None, top_k: int | None = None, use_llm: bool = True) -> QueryResponse:
        if not session_id:
            response = self.query_service.ask(query=query, top_k=top_k, use_llm=use_llm)
            response.rewritten_query = query
            response.resolved_company = str(response.intent.target_company or "")
            response.resolved_profile = ""
            response.used_history = False
            return response

        state = self.store.get(session_id)
        resolved = resolve_query(query, state)
        response = self.query_service.ask(query=resolved.rewritten_query, top_k=top_k, use_llm=use_llm)
        response.rewritten_query = resolved.rewritten_query
        response.resolved_company = resolved.resolved_company or str(response.intent.target_company or "")
        response.resolved_profile = resolved.resolved_profile_id
        response.used_history = resolved.used_history

        updated_state = self._ensure_state(session_id, state)
        updated_state.current_company = response.resolved_company or updated_state.current_company
        updated_state.current_profile_id = response.resolved_profile or updated_state.current_profile_id
        updated_state.current_question_type = resolved.question_type or updated_state.current_question_type
        updated_state.current_subject = resolved.current_subject or updated_state.current_subject
        updated_state.current_topic = resolved.current_subject or updated_state.current_topic
        updated_state.last_rewritten_query = resolved.rewritten_query
        updated_state.last_answer_summary = self._build_answer_summary(response)
        updated_state.last_citations = self._build_citation_summaries(response)
        updated_state.updated_at = datetime.now(timezone.utc).isoformat()
        updated_state.history_turns.append(
            ConversationTurn(
                query=query,
                rewritten_query=resolved.rewritten_query,
                answer_summary=updated_state.last_answer_summary,
                resolved_company=updated_state.current_company,
                resolved_profile_id=updated_state.current_profile_id,
                question_type=updated_state.current_question_type,
                current_subject=updated_state.current_subject,
                used_history=resolved.used_history,
                rewrite_reason=resolved.rewrite_reason,
                citations=updated_state.last_citations,
            )
        )
        updated_state.history_turns = updated_state.history_turns[-self.history_limit :]
        self.store.save(updated_state)
        return response

    def clear(self, session_id: str) -> None:
        self.store.delete(session_id)

    def get_state(self, session_id: str) -> ConversationState | None:
        if not session_id:
            return None
        return self.store.get(session_id)
