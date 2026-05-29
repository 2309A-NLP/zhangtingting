from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, log_timed, preview_text
from app.db.milvus_client import get_milvus_client
from app.knowledge.embedder import BgeM3Embedder
from app.retriever.models import RetrievalBundle, RetrievedChunk
from app.retriever.query_rewrite import QueryRewriter
from app.retriever.reranker import BgeReranker

logger = get_logger(__name__)
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


class HybridRetriever:
    def __init__(
        self,
        *,
        embedder: BgeM3Embedder | None = None,
        query_rewriter: QueryRewriter | None = None,
        reranker: BgeReranker | None = None,
    ) -> None:
        self.settings = get_settings()
        self.logger = logger
        self.embedder = embedder or BgeM3Embedder()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.reranker = reranker or BgeReranker()
        self.milvus_client = get_milvus_client()

    @log_timed("hybrid_retrieve")
    async def retrieve(
        self,
        *,
        user_id: str,
        role_id: str,
        query: str,
        role_category: str = "general",
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        candidate_pool_size: int = 2000,
    ) -> RetrievalBundle:
        effective_top_k = top_k or self.settings.retrieval_top_k
        fusion_candidate_k = max(
            effective_top_k,
            self.settings.retrieval_vector_top_k,
            self.settings.retrieval_bm25_top_k,
            self.settings.rerank_top_k,
            effective_top_k * 4,
        )
        emit_runtime_trace(
            self.logger,
            "hybrid_retrieval_entered",
            user_id=user_id,
            role_id=role_id,
            role_category=role_category,
            query=query,
            history_count=len(history or []),
            top_k=effective_top_k,
            candidate_pool_size=candidate_pool_size,
            fusion_candidate_k=fusion_candidate_k,
        )

        rewrite_result = await self.query_rewriter.rewrite(query=query, history=history or [])
        target_query = rewrite_result.rewritten_query
        emit_runtime_trace(
            self.logger,
            "hybrid_retrieval_query_ready",
            original_query=query,
            rewritten_query=target_query,
            rewrite_reason=rewrite_result.reason,
        )

        dense_results = await self._dense_search(
            user_id=user_id,
            role_id=role_id,
            query=target_query,
            top_k=self.settings.retrieval_vector_top_k,
            role_category=role_category,
        )
        emit_runtime_trace(
            self.logger,
            "hybrid_retrieval_dense_done",
            dense_count=len(dense_results),
            dense_preview=[preview_text(item.text, 80) for item in dense_results[:3]],
        )

        bm25_results = await self._bm25_search(
            user_id=user_id,
            role_id=role_id,
            query=target_query,
            role_category=role_category,
            candidate_pool_size=max(candidate_pool_size, self.settings.retrieval_bm25_top_k),
        )
        emit_runtime_trace(
            self.logger,
            "hybrid_retrieval_bm25_done",
            bm25_count=len(bm25_results),
            bm25_preview=[preview_text(item.text, 80) for item in bm25_results[:3]],
        )

        fused = self._rrf_fuse([dense_results, bm25_results], top_k=fusion_candidate_k)
        emit_runtime_trace(
            self.logger,
            "hybrid_retrieval_rrf_done",
            fused_count=len(fused),
            fused_preview=[preview_text(item.text, 80) for item in fused[:3]],
        )

        reranked = await self.reranker.rerank(
            query=target_query,
            candidates=fused,
            top_k=effective_top_k,
        )
        emit_runtime_trace(
            self.logger,
            "hybrid_retrieval_rerank_done",
            reranked_count=len(reranked),
            reranked_preview=[preview_text(item.text, 80) for item in reranked[:3]],
        )

        logger.info(
            "hybrid_retrieval_completed",
            user_id=user_id,
            role_id=role_id,
            query=query,
            rewritten_query=target_query,
            dense_count=len(dense_results),
            bm25_count=len(bm25_results),
            final_count=len(reranked),
        )
        return RetrievalBundle(
            query=query,
            rewritten_query=target_query,
            dense_results=dense_results,
            bm25_results=bm25_results,
            fused_results=reranked,
        )

    @log_timed("dense_search", emit_start=False)
    async def _dense_search(
        self,
        *,
        user_id: str,
        role_id: str,
        query: str,
        top_k: int,
        role_category: str,
    ) -> list[RetrievedChunk]:
        emit_runtime_trace(
            self.logger,
            "dense_search_prepare_embedding",
            query=query,
            top_k=top_k,
            role_category=role_category,
        )
        embedded_query = await self.embedder.embed(
            [
                self._build_query_chunk(
                    user_id=user_id,
                    role_id=role_id,
                    query=query,
                    role_category=role_category,
                )
            ]
        )
        vector = embedded_query[0].embedding
        filter_expr = self._build_filter(user_id=user_id, role_id=role_id, role_category=role_category)
        emit_runtime_trace(
            self.logger,
            "dense_search_calling_milvus",
            filter_expr=filter_expr,
            vector_dim=len(vector),
            vector_preview=vector[:8],
            collection_name=self.settings.milvus_collection_name,
        )

        raw_results = await asyncio.to_thread(
            self.milvus_client.search,
            collection_name=self.settings.milvus_collection_name,
            data=[vector],
            anns_field="embedding",
            filter=filter_expr,
            limit=top_k,
            output_fields=["id", "doc_id", "chunk_id", "text", "source", "role_category"],
            search_params={
                "metric_type": self.settings.milvus_metric_type,
                "params": {"nprobe": self.settings.milvus_search_nprobe},
            },
        )
        hits = raw_results[0] if raw_results else []
        return [self._build_retrieved_chunk(hit, retrieval_type="dense") for hit in hits]

    @log_timed("bm25_search", emit_start=False)
    async def _bm25_search(
        self,
        *,
        user_id: str,
        role_id: str,
        query: str,
        role_category: str,
        candidate_pool_size: int,
    ) -> list[RetrievedChunk]:
        filter_expr = self._build_filter(user_id=user_id, role_id=role_id, role_category=role_category)
        emit_runtime_trace(
            self.logger,
            "bm25_search_query_candidates",
            filter_expr=filter_expr,
            candidate_pool_size=candidate_pool_size,
            query=query,
        )
        candidates = await asyncio.to_thread(
            self.milvus_client.query,
            collection_name=self.settings.milvus_collection_name,
            filter=filter_expr,
            output_fields=["id", "doc_id", "chunk_id", "text", "source", "role_category"],
            limit=candidate_pool_size,
        )
        if not candidates:
            emit_runtime_trace(self.logger, "bm25_search_empty_candidates")
            return []

        tokenized_docs = [self._tokenize(item["text"]) for item in candidates]
        bm25 = BM25Okapi(tokenized_docs)
        query_tokens = self._tokenize(query)
        emit_runtime_trace(
            self.logger,
            "bm25_search_tokenized",
            query_tokens=query_tokens[:20],
            candidate_count=len(candidates),
        )
        scores = bm25.get_scores(query_tokens)

        scored_results = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )[: self.settings.retrieval_bm25_top_k]

        results: list[RetrievedChunk] = []
        for entity, score in scored_results:
            results.append(
                RetrievedChunk(
                    id=str(entity["id"]),
                    doc_id=str(entity["doc_id"]),
                    chunk_id=str(entity["chunk_id"]),
                    text=str(entity["text"]),
                    source=str(entity.get("source", "")),
                    role_category=str(entity.get("role_category", role_category)),
                    score=float(score),
                    retrieval_type="bm25",
                    metadata={"filter": filter_expr},
                )
            )

        emit_runtime_trace(
            self.logger,
            "bm25_search_scored",
            result_count=len(results),
            result_preview=[preview_text(item.text, 80) for item in results[:3]],
        )
        return results

    def _rrf_fuse(self, result_sets: list[list[RetrievedChunk]], top_k: int) -> list[RetrievedChunk]:
        fused_scores: dict[str, float] = defaultdict(float)
        exemplar: dict[str, RetrievedChunk] = {}

        for result_set in result_sets:
            for rank, chunk in enumerate(result_set, start=1):
                fused_scores[chunk.id] += 1.0 / (self.settings.retrieval_rrf_k + rank)
                exemplar[chunk.id] = chunk

        fused = [
            RetrievedChunk(
                id=chunk_id,
                doc_id=exemplar[chunk_id].doc_id,
                chunk_id=exemplar[chunk_id].chunk_id,
                text=exemplar[chunk_id].text,
                source=exemplar[chunk_id].source,
                role_category=exemplar[chunk_id].role_category,
                score=score,
                retrieval_type="rrf",
                heading_path=exemplar[chunk_id].heading_path,
                metadata=exemplar[chunk_id].metadata,
            )
            for chunk_id, score in fused_scores.items()
        ]
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[:top_k]

    def _build_filter(self, *, user_id: str, role_id: str, role_category: str) -> str:
        tenant_keys = [f"{user_id}:{role_id}"]
        shared_tenant_key = f"{self.settings.shared_preset_user_id}:{role_id}"
        if shared_tenant_key not in tenant_keys:
            tenant_keys.append(shared_tenant_key)

        if len(tenant_keys) == 1:
            tenant_clause = f'tenant_key == "{tenant_keys[0]}"'
        else:
            tenant_clause = "(" + " or ".join(f'tenant_key == "{key}"' for key in tenant_keys) + ")"

        clauses = [tenant_clause]
        if role_category and role_category != "general":
            clauses.append(f'role_category == "{role_category}"')
        return " and ".join(clauses)

    def _build_retrieved_chunk(self, hit: Any, retrieval_type: str) -> RetrievedChunk:
        entity = hit.get("entity", {}) if isinstance(hit, dict) else getattr(hit, "entity", {})
        if not entity and hasattr(hit, "fields"):
            entity = hit.fields

        return RetrievedChunk(
            id=str(entity.get("id")),
            doc_id=str(entity.get("doc_id")),
            chunk_id=str(entity.get("chunk_id")),
            text=str(entity.get("text", "")),
            source=str(entity.get("source", "")),
            role_category=str(entity.get("role_category", "general")),
            score=float(hit.get("distance", 0.0) if isinstance(hit, dict) else getattr(hit, "distance", 0.0)),
            retrieval_type=retrieval_type,
            metadata={"raw_hit": str(hit)},
        )

    def _build_query_chunk(self, *, user_id: str, role_id: str, query: str, role_category: str):
        from app.knowledge.models import ChunkedDocument

        return ChunkedDocument(
            id="query-embedding",
            doc_id="query",
            chunk_id="query",
            user_id=user_id,
            role_id=role_id,
            role_category=role_category,
            text=query,
            token_count=len(self._tokenize(query)),
            chunk_index=0,
            source="query",
            heading_path="query",
            metadata={},
        )

    def _tokenize(self, text: str) -> list[str]:
        tokens = TOKEN_PATTERN.findall(text.lower())
        if tokens:
            return tokens
        return list(text.lower())
