from __future__ import annotations

import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.models import BuiltContext, ContextSource
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, log_timed, preview_text
from app.db.redis_client import chat_recent_key, get_redis, memory_summary_key
from app.retriever.hybrid import HybridRetriever
from app.retriever.models import RetrievalBundle, RetrievedChunk

logger = get_logger(__name__)
MIN_CONTEXT_SOURCE_SCORE = 0.5
MAX_CONTEXT_SOURCE_COUNT = 5
EVIDENCE_RESPONSE_RULES = (
    "回答规则："
    "优先直接回答用户问题。"
    "如果已有检索证据，只能基于这些证据作答，不要补充证据之外的常见方案或百科知识。"
    "如果证据不足，明确说明现有资料不足。"
    "除非用户要求详细展开，否则控制在简洁回答加少量依据。"
)


class ContextBuilder:
    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.settings = get_settings()
        self.logger = logger
        self.retriever = retriever or HybridRetriever()
        self.redis = get_redis()

    @log_timed("context_build")
    async def build(
        self,
        *,
        db_session: AsyncSession,
        user_id: str,
        role_id: str,
        role_name: str,
        role_category: str,
        system_prompt: str,
        query: str,
        session_id: str | None = None,
        top_k: int | None = None,
    ) -> BuiltContext:
        emit_runtime_trace(
            self.logger,
            "context_build_entered",
            user_id=user_id,
            role_id=role_id,
            role_name=role_name,
            role_category=role_category,
            session_id=session_id or "",
            query=query,
            top_k=top_k or self.settings.retrieval_top_k,
        )

        recent_memory = await self._load_recent_memory(user_id=user_id, role_id=role_id, session_id=session_id)
        long_memory = await self._load_long_memory(user_id=user_id, role_id=role_id, session_id=session_id)
        emit_runtime_trace(
            self.logger,
            "context_memory_loaded",
            recent_memory_count=len(recent_memory),
            long_memory_length=len(long_memory),
            long_memory_preview=long_memory,
        )

        history_rows = (
            []
            if session_id
            else await self._load_history_from_mysql(db_session=db_session, user_id=user_id, role_id=role_id)
        )
        combined_history = recent_memory or history_rows
        emit_runtime_trace(
            self.logger,
            "context_history_prepared",
            mysql_history_count=len(history_rows),
            combined_history_count=len(combined_history),
            combined_history_preview=preview_text(combined_history[-1]["content"], 100) if combined_history else "",
        )

        retrieval_top_k = top_k if top_k is not None else self.settings.retrieval_top_k
        if self.settings.retrieval_enabled and retrieval_top_k > 0:
            retrieval_bundle = await self.retriever.retrieve(
                user_id=user_id,
                role_id=role_id,
                query=query,
                role_category=role_category,
                history=combined_history,
                top_k=retrieval_top_k,
            )
        else:
            logger.info(
                "retrieval_skipped",
                user_id=user_id,
                role_id=role_id,
                query=query,
                retrieval_enabled=self.settings.retrieval_enabled,
                top_k=retrieval_top_k,
            )
            retrieval_bundle = RetrievalBundle(
                query=query,
                rewritten_query=query,
                dense_results=[],
                bm25_results=[],
                fused_results=[],
            )

        filtered_results = self._filter_context_results(retrieval_bundle.fused_results)
        context_sources = [
            ContextSource(
                doc_id=item.doc_id,
                chunk_id=item.chunk_id,
                source=item.source,
                score=item.score,
                text=item.text,
            )
            for item in filtered_results
        ]
        emit_runtime_trace(
            self.logger,
            "context_sources_selected",
            fused_count=len(retrieval_bundle.fused_results),
            selected_count=len(context_sources),
            selected_preview=[preview_text(item.text, 80) for item in context_sources[:3]],
        )

        context_block = self._render_context_block(context_sources)
        memory_block = self._render_memory_block(long_memory, combined_history)
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "system", "content": EVIDENCE_RESPONSE_RULES})
        if memory_block:
            messages.append({"role": "system", "content": memory_block})
        if context_block:
            messages.append({"role": "system", "content": context_block})
        messages.extend(combined_history[-self.settings.chat_recent_rounds * 2 :])
        messages.append({"role": "user", "content": query})

        logger.info(
            "context_built",
            user_id=user_id,
            role_id=role_id,
            role_name=role_name,
            query=query,
            rewritten_query=retrieval_bundle.rewritten_query,
            source_count=len(context_sources),
        )
        emit_runtime_trace(
            self.logger,
            "context_messages_ready",
            message_count=len(messages),
            system_prompt_preview=system_prompt,
            context_block_preview=context_block,
        )

        return BuiltContext(
            messages=messages,
            context_sources=context_sources,
            rewritten_query=retrieval_bundle.rewritten_query,
            retrieval_debug={
                "dense_count": len(retrieval_bundle.dense_results),
                "bm25_count": len(retrieval_bundle.bm25_results),
                "fused_count": len(retrieval_bundle.fused_results),
                "cited_count": len(context_sources),
            },
        )

    def _filter_context_results(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not results:
            return []

        # RRF/rerank scores are not on the same scale as dense cosine similarity.
        # Applying a hard 0.5 threshold to fused results drops all retrieved context.
        filtered: list[RetrievedChunk] = []
        for item in results:
            if not item.text.strip():
                continue
            if item.retrieval_type == "dense" and item.score < MIN_CONTEXT_SOURCE_SCORE:
                continue
            filtered.append(item)

        return filtered[:MAX_CONTEXT_SOURCE_COUNT]

    async def _load_recent_memory(
        self,
        *,
        user_id: str,
        role_id: str,
        session_id: str | None = None,
    ) -> list[dict[str, str]]:
        key = chat_recent_key(user_id, role_id, session_id)
        items = await self.redis.lrange(key, 0, self.settings.chat_recent_rounds * 2 - 1)
        messages: list[dict[str, str]] = []
        for item in items:
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                continue

            role = parsed.get("role", "").strip()
            content = parsed.get("content", "").strip()
            if role and content:
                messages.append({"role": role, "content": content})
        return messages

    async def _load_long_memory(self, *, user_id: str, role_id: str, session_id: str | None = None) -> str:
        key = memory_summary_key(user_id, role_id, session_id)
        summary = await self.redis.get(key)
        return summary.strip() if summary else ""

    async def _load_history_from_mysql(
        self,
        *,
        db_session: AsyncSession,
        user_id: str,
        role_id: str,
    ) -> list[dict[str, str]]:
        sql = text(
            """
            SELECT query, response
            FROM conversations
            WHERE user_id = :user_id AND role_id = :role_id
            ORDER BY timestamp DESC
            LIMIT :limit_count
            """
        )
        result = await db_session.execute(
            sql,
            {
                "user_id": user_id,
                "role_id": role_id,
                "limit_count": self.settings.chat_history_load_limit,
            },
        )
        rows = result.mappings().all()
        history: list[dict[str, str]] = []
        for row in reversed(rows):
            if row["query"]:
                history.append({"role": "user", "content": row["query"]})
            if row["response"]:
                history.append({"role": "assistant", "content": row["response"]})
        return history

    def _render_context_block(self, context_sources: list[ContextSource]) -> str:
        if not context_sources:
            return ""

        blocks = []
        for index, item in enumerate(context_sources, start=1):
            blocks.append(
                f"[Evidence {index}] doc_id={item.doc_id} chunk_id={item.chunk_id} source={item.source}\n{item.text}"
            )
        return (
            "Retrieved knowledge context is provided below. Prioritize these grounded facts in your answer.\n\n"
            + "\n\n".join(blocks)
        )

    def _render_memory_block(self, long_memory: str, recent_history: list[dict[str, str]]) -> str:
        parts: list[str] = []
        if long_memory:
            parts.append(f"Long-term memory summary:\n{long_memory}")
        if recent_history:
            recent_lines = [
                f"{item['role']}: {item['content']}"
                for item in recent_history[-6:]
                if item.get("content")
            ]
            if recent_lines:
                parts.append("Recent conversation snippets:\n" + "\n".join(recent_lines))
        return "\n\n".join(parts)
