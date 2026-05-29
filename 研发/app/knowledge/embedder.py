from __future__ import annotations

import asyncio
import threading
from math import ceil
from pathlib import Path
from typing import Literal

import torch
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, log_timed, preview_text
from app.knowledge.models import ChunkedDocument, EmbeddedChunk

logger = get_logger(__name__)

try:
    from FlagEmbedding import BGEM3FlagModel
except Exception:  # pragma: no cover - optional dependency fallback
    BGEM3FlagModel = None  # type: ignore[assignment]


class BgeM3Embedder:
    _shared_model: SentenceTransformer | object | None = None
    _shared_backend: Literal["bge_m3", "sentence_transformer"] | None = None
    _shared_model_key: tuple[str, str, bool] | None = None
    _shared_model_lock = threading.Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logger
        self._model: SentenceTransformer | object | None = None
        self._backend: Literal["bge_m3", "sentence_transformer"] | None = None

    @log_timed("embed_chunks")
    async def embed(self, chunks: list[ChunkedDocument]) -> list[EmbeddedChunk]:
        if not chunks:
            emit_runtime_trace(self.logger, "embed_skipped", reason="empty_chunks")
            return []

        emit_runtime_trace(
            self.logger,
            "embed_entered",
            chunk_count=len(chunks),
            first_chunk=preview_text(chunks[0].text, 120),
        )
        return await asyncio.to_thread(self._embed_sync, chunks)

    def _get_model(self) -> SentenceTransformer | object:
        model_ref = self._resolve_model_ref()
        should_use_sentence_transformer = self._should_use_sentence_transformer(model_ref)
        model_key = (
            model_ref,
            self.settings.embedding_device,
            should_use_sentence_transformer,
        )

        if (
            self.__class__._shared_model is not None
            and self.__class__._shared_model_key == model_key
        ):
            self._model = self.__class__._shared_model
            self._backend = self.__class__._shared_backend
            return self._model

        with self.__class__._shared_model_lock:
            if (
                self.__class__._shared_model is not None
                and self.__class__._shared_model_key == model_key
            ):
                self._model = self.__class__._shared_model
                self._backend = self.__class__._shared_backend
                return self._model

            if should_use_sentence_transformer:
                backend: Literal["bge_m3", "sentence_transformer"] = "sentence_transformer"
                model = SentenceTransformer(
                    model_ref,
                    device=self.settings.embedding_device,
                )
            else:
                if BGEM3FlagModel is None:
                    logger.warning(
                        "flag_embedding_unavailable_fallback",
                        model_ref=model_ref,
                        fallback_backend="sentence_transformer",
                    )
                    backend = "sentence_transformer"
                    model = SentenceTransformer(
                        model_ref,
                        device=self.settings.embedding_device,
                    )
                else:
                    backend = "bge_m3"
                    model = BGEM3FlagModel(
                        model_ref,
                        use_fp16=self.settings.embedding_device.startswith("cuda"),
                        device=self.settings.embedding_device,
                    )

            self.__class__._shared_model = model
            self.__class__._shared_backend = backend
            self.__class__._shared_model_key = model_key
            self._model = model
            self._backend = backend
            logger.info("embedding_backend_selected", backend=self._backend, model_ref=model_ref)
            emit_runtime_trace(
                self.logger,
                "embedding_model_loaded",
                backend=self._backend,
                model_ref=model_ref,
                device=self.settings.embedding_device,
            )
        return self._model

    def _resolve_model_ref(self) -> str:
        model_path = self.settings.embedding_model_path.strip()
        if model_path:
            path = Path(model_path)
            if path.exists():
                logger.info("embedding_model_path_selected", model_path=str(path))
                return str(path)
        return self.settings.embedding_model_name

    def _should_use_sentence_transformer(self, model_ref: str) -> bool:
        ref_lower = model_ref.lower()
        if "m3e" in ref_lower:
            return True

        path = Path(model_ref)
        if path.exists():
            has_bge_m3_sidecars = (path / "colbert_linear.pt").exists() and (path / "sparse_linear.pt").exists()
            return not has_bge_m3_sidecars

        return False

    def _embed_sync(self, chunks: list[ChunkedDocument]) -> list[EmbeddedChunk]:
        model = self._get_model()
        texts = [chunk.text for chunk in chunks]
        batch_size = self._calculate_batch_size(len(texts))
        emit_runtime_trace(
            self.logger,
            "embed_sync_started",
            backend=self._backend,
            chunk_count=len(chunks),
            batch_size=batch_size,
        )

        outputs: list[EmbeddedChunk] = []
        for batch_index in range(ceil(len(texts) / batch_size)):
            batch_chunks = chunks[batch_index * batch_size : (batch_index + 1) * batch_size]
            batch_texts = [chunk.text for chunk in batch_chunks]
            emit_runtime_trace(
                self.logger,
                "embed_batch_started",
                batch_index=batch_index,
                batch_count=len(batch_chunks),
                batch_preview=preview_text(batch_texts[0], 100) if batch_texts else "",
            )

            if self._backend == "sentence_transformer":
                dense_vectors = model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            else:
                dense_vectors = model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    max_length=8192,
                )["dense_vecs"]

            for chunk, vector in zip(batch_chunks, dense_vectors, strict=True):
                outputs.append(
                    EmbeddedChunk(
                        id=chunk.id,
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        user_id=chunk.user_id,
                        role_id=chunk.role_id,
                        role_category=chunk.role_category,
                        text=chunk.text,
                        embedding=[float(value) for value in vector.tolist()],
                        source=chunk.source,
                        heading_path=chunk.heading_path,
                        metadata=chunk.metadata,
                    )
                )

            emit_runtime_trace(
                self.logger,
                "embed_batch_finished",
                batch_index=batch_index,
                accumulated_count=len(outputs),
                vector_dim=len(outputs[-1].embedding) if outputs else 0,
                vector_preview=outputs[-1].embedding[:8] if outputs else [],
            )

        logger.info("chunks_embedded", chunk_count=len(outputs), batch_size=batch_size)
        emit_runtime_trace(
            self.logger,
            "embed_completed",
            chunk_count=len(outputs),
            batch_size=batch_size,
            last_vector_preview=outputs[-1].embedding[:8] if outputs else [],
        )
        return outputs

    def _calculate_batch_size(self, chunk_count: int) -> int:
        minimum = self.settings.embedding_batch_size_min
        maximum = self.settings.embedding_batch_size_max

        if self.settings.embedding_device.startswith("cuda") and torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            utilization = free_memory / max(total_memory, 1)
            if utilization > 0.5:
                return min(maximum, chunk_count)
            if utilization > 0.25:
                return min(max(minimum * 2, maximum), chunk_count)
            return min(minimum, chunk_count)

        return min(minimum, chunk_count)
