from __future__ import annotations

from datetime import datetime, timezone

from backend.conversation.models import CitationSummary, ConversationState, ConversationTurn, ResolvedQuery
from backend.conversation.query_understanding import resolve_query
from backend.conversation.store import ConversationStore
from backend.retrieval.unified_query_service import UnifiedDefaultQueryService
from backend.schemas import QueryResponse
from backend.services.rag_pipeline import RAGPipeline


class _ConversationSessionMixin:
    def __init__(self, store: ConversationStore, history_limit: int = 6) -> None:
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

    def _prepare_state(
        self,
        session_id: str,
        state: ConversationState | None,
        *,
        corpus: str,
        upload_id: str = "",
    ) -> ConversationState:
        normalized_corpus = str(corpus or "default").strip() or "default"
        normalized_upload_id = str(upload_id or "").strip()

        if state is not None:
            current_corpus = str(state.current_corpus or "default").strip() or "default"
            current_upload_id = str(state.current_upload_id or "").strip()
            upload_changed = normalized_corpus == "uploaded" and normalized_upload_id and current_upload_id not in {"", normalized_upload_id}
            if current_corpus != normalized_corpus or upload_changed:
                state = None

        prepared = self._ensure_state(session_id, state)
        prepared.current_corpus = normalized_corpus
        prepared.current_upload_id = normalized_upload_id if normalized_corpus == "uploaded" else ""
        return prepared

    def _finalize_response(
        self,
        response: QueryResponse,
        *,
        original_query: str,
        resolved: ResolvedQuery | None = None,
    ) -> QueryResponse:
        response.rewritten_query = resolved.rewritten_query if resolved is not None else original_query
        response.resolved_company = (
            resolved.resolved_company if (resolved is not None and resolved.resolved_company) else str(response.intent.target_company or "")
        )
        response.resolved_profile = resolved.resolved_profile_id if resolved is not None else ""
        response.used_history = bool(resolved.used_history) if resolved is not None else False
        return response

    def _save_state(
        self,
        session_id: str,
        state: ConversationState,
        *,
        query: str,
        resolved: ResolvedQuery,
        response: QueryResponse,
        corpus: str,
        upload_id: str = "",
    ) -> None:
        state.current_corpus = str(corpus or "default").strip() or "default"
        state.current_upload_id = str(upload_id or "").strip() if state.current_corpus == "uploaded" else ""
        state.current_company = response.resolved_company or state.current_company
        state.current_profile_id = response.resolved_profile or state.current_profile_id
        state.current_question_type = resolved.question_type or state.current_question_type
        state.current_subject = resolved.current_subject or state.current_subject
        state.current_topic = resolved.current_subject or state.current_topic
        state.last_rewritten_query = resolved.rewritten_query
        state.last_answer_summary = self._build_answer_summary(response)
        state.last_citations = self._build_citation_summaries(response)
        state.updated_at = datetime.now(timezone.utc).isoformat()
        state.history_turns.append(
            ConversationTurn(
                query=query,
                rewritten_query=resolved.rewritten_query,
                answer_summary=state.last_answer_summary,
                resolved_company=state.current_company,
                resolved_profile_id=state.current_profile_id,
                corpus=state.current_corpus,
                upload_id=state.current_upload_id,
                question_type=state.current_question_type,
                current_subject=state.current_subject,
                used_history=resolved.used_history,
                rewrite_reason=resolved.rewrite_reason,
                citations=state.last_citations,
            )
        )
        state.history_turns = state.history_turns[-self.history_limit :]
        self.store.save(state)

    def clear(self, session_id: str) -> None:
        self.store.delete(session_id)

    def get_state(self, session_id: str) -> ConversationState | None:
        if not session_id:
            return None
        return self.store.get(session_id)


class ConversationQueryService(_ConversationSessionMixin):
    def __init__(self, query_service: UnifiedDefaultQueryService, store: ConversationStore, history_limit: int = 6) -> None:
        super().__init__(store=store, history_limit=history_limit)
        self.query_service = query_service

    def ask(self, query: str, session_id: str | None, top_k: int | None = None, use_llm: bool = True) -> QueryResponse:
        if not session_id:
            response = self.query_service.ask(query=query, top_k=top_k, use_llm=use_llm)
            return self._finalize_response(response, original_query=query)

        state = self.store.get(session_id)
        prepared_state = self._prepare_state(session_id, state, corpus="default")
        resolved = resolve_query(query, prepared_state)
        response = self.query_service.ask(query=resolved.rewritten_query, top_k=top_k, use_llm=use_llm)
        self._finalize_response(response, original_query=query, resolved=resolved)
        self._save_state(session_id, prepared_state, query=query, resolved=resolved, response=response, corpus="default")
        return response


class UploadedConversationQueryService(_ConversationSessionMixin):
    def __init__(self, pipeline: RAGPipeline, store: ConversationStore, history_limit: int = 6) -> None:
        super().__init__(store=store, history_limit=history_limit)
        self.pipeline = pipeline

    def ask(
        self,
        query: str,
        session_id: str | None,
        upload_id: str | None,
        top_k: int | None = None,
        use_llm: bool = True,
    ) -> QueryResponse:
        if not session_id:
            response = self.pipeline.ask(
                query=query,
                top_k=top_k,
                use_llm=use_llm,
                corpus="uploaded",
                upload_id=upload_id or None,
            )
            return self._finalize_response(response, original_query=query)

        state = self.store.get(session_id)
        effective_upload_id = str(upload_id or (state.current_upload_id if state is not None else "") or "").strip()
        prepared_state = self._prepare_state(
            session_id,
            state,
            corpus="uploaded",
            upload_id=effective_upload_id,
        )
        resolved = resolve_query(query, prepared_state)
        response = self.pipeline.ask(
            query=resolved.rewritten_query,
            top_k=top_k,
            use_llm=use_llm,
            corpus="uploaded",
            upload_id=effective_upload_id or None,
        )
        self._finalize_response(response, original_query=query, resolved=resolved)
        self._save_state(
            session_id,
            prepared_state,
            query=query,
            resolved=resolved,
            response=response,
            corpus="uploaded",
            upload_id=effective_upload_id,
        )
        return response
