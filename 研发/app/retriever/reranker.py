from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, log_timed, preview_text
from app.retriever.models import RetrievedChunk

logger = get_logger(__name__)

try:
    from FlagEmbedding import FlagReranker
except Exception:  # pragma: no cover - optional dependency fallback
    FlagReranker = None  # type: ignore[assignment]


class BgeReranker:
    _shared_model: object | None = None
    _shared_model_key: tuple[str, str, bool] | None = None
    _shared_model_lock = threading.Lock()
    _shared_infer_lock = threading.Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logger
        self._model: object | None = None

    @log_timed("rerank")
    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
        emit_runtime_trace(
            self.logger,
            "rerank_entered",
            query=query,
            candidate_count=len(candidates),
            top_k=top_k or self.settings.rerank_top_k,
            enabled=self.settings.retrieval_enable_rerank,
        )
        if not candidates or not self.settings.retrieval_enable_rerank:
            emit_runtime_trace(self.logger, "rerank_skipped", reason="disabled_or_empty")
            return candidates[: top_k or len(candidates)]
        if FlagReranker is None:
            logger.warning("rerank_backend_unavailable_skip", reason="flag_embedding_import_failed")
            emit_runtime_trace(self.logger, "rerank_skipped", reason="flag_embedding_unavailable")
            return candidates[: top_k or len(candidates)]
        return await asyncio.to_thread(self._rerank_sync, query, candidates, top_k)

    def _get_model(self) -> object:
        use_fp16 = self.settings.rerank_device.startswith("cuda")
        model_ref = self._resolve_model_ref()
        model_key = (model_ref, self.settings.rerank_device, use_fp16)

        if (
            self.__class__._shared_model is not None
            and self.__class__._shared_model_key == model_key
        ):
            self._model = self.__class__._shared_model
            return self._model

        with self.__class__._shared_model_lock:
            if (
                self.__class__._shared_model is not None
                and self.__class__._shared_model_key == model_key
            ):
                self._model = self.__class__._shared_model
                return self._model

            use_fp16 = self.settings.rerank_device.startswith("cuda")
            model = FlagReranker(
                model_ref,
                use_fp16=use_fp16,
                devices=self.settings.rerank_device,
            )
            self.__class__._shared_model = model
            self.__class__._shared_model_key = model_key
            self._model = model
            emit_runtime_trace(
                self.logger,
                "rerank_model_loaded",
                model_ref=model_ref,
                device=self.settings.rerank_device,
                use_fp16=use_fp16,
            )
        return self._model

    def _resolve_model_ref(self) -> str:
        model_path = self.settings.rerank_model_path.strip()
        if model_path:
            path = Path(model_path)
            if path.exists():
                logger.info("rerank_model_path_selected", model_path=str(path))
                return str(path)
        return self.settings.rerank_model_name

    def _rerank_sync(self, query: str, candidates: list[RetrievedChunk], top_k: int | None) -> list[RetrievedChunk]:
        model = self._get_model()
        pairs = [[query, item.text] for item in candidates]
        emit_runtime_trace(
            self.logger,
            "rerank_scoring_pairs",
            pair_count=len(pairs),
            first_candidate=preview_text(candidates[0].text, 100) if candidates else "",
        )

        # FlagEmbedding internally moves the model to the target device during scoring.
        # Under concurrent requests this is not thread-safe, so we serialize scoring.
        with self.__class__._shared_infer_lock:
            scores = model.compute_score(pairs, normalize=True)

        reranked = [
            RetrievedChunk(
                id=item.id,
                doc_id=item.doc_id,
                chunk_id=item.chunk_id,
                text=item.text,
                source=item.source,
                role_category=item.role_category,
                score=float(score),
                retrieval_type="rerank",
                heading_path=item.heading_path,
                metadata=item.metadata,
            )
            for item, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        limit = top_k or self.settings.rerank_top_k
        logger.info("candidates_reranked", candidate_count=len(candidates), output_count=min(limit, len(reranked)))
        emit_runtime_trace(
            self.logger,
            "rerank_scored",
            output_count=min(limit, len(reranked)),
            top_scores=[round(item.score, 4) for item in reranked[:5]],
            top_preview=[preview_text(item.text, 80) for item in reranked[:3]],
        )
        return reranked[:limit]
