from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from pymilvus import MilvusClient

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import emit_runtime_trace, log_timed, preview_text
from app.db.milvus_client import get_milvus_client
from app.db.mysql_client import get_mysql_session_factory
from app.db.redis_client import get_redis, ingest_status_key
from app.knowledge.chunker import SemanticChunker
from app.knowledge.cleaner import DocumentCleaner
from app.knowledge.embedder import BgeM3Embedder
from app.knowledge.loader import DocumentLoader
from app.knowledge.models import EmbeddedChunk, RawDocument
from app.services.audit_service import AuditService
from app.services.knowledge_file_service import KnowledgeFileService
from app.storage.minio_service import MinioStorageService

logger = get_logger(__name__)


class KnowledgeIngestService:
    def __init__(
        self,
        loader: DocumentLoader | None = None,
        cleaner: DocumentCleaner | None = None,
        chunker: SemanticChunker | None = None,
        embedder: BgeM3Embedder | None = None,
        milvus_client: MilvusClient | None = None,
        storage_service: MinioStorageService | None = None,
    ) -> None:
        self.logger = logger
        self.loader = loader or DocumentLoader()
        self.cleaner = cleaner or DocumentCleaner()
        self.chunker = chunker or SemanticChunker()
        self.embedder = embedder or BgeM3Embedder()
        self.milvus_client = milvus_client or get_milvus_client()
        self.storage_service = storage_service or MinioStorageService()

    @log_timed("knowledge_ingest")
    async def ingest_document(
        self,
        raw_document: RawDocument,
        *,
        role_category: str = "general",
        mode: str = "incremental",
        collection_name: str | None = None,
        replace_doc_id: str | None = None,
    ) -> dict[str, Any]:
        redis = get_redis()
        settings = get_settings()
        session_factory = get_mysql_session_factory()
        status_key = ingest_status_key(raw_document.user_id, raw_document.role_id, raw_document.task_id)
        started_at = int(time.time())

        emit_runtime_trace(
            self.logger,
            "knowledge_ingest_entered",
            task_id=raw_document.task_id,
            doc_id=raw_document.file_id,
            user_id=raw_document.user_id,
            role_id=raw_document.role_id,
            role_category=role_category,
            mode=mode,
            collection_name=collection_name or settings.milvus_collection_name,
            file_name=raw_document.file_name,
            source_uri=raw_document.source_uri,
        )

        await redis.hset(
            status_key,
            mapping={
                "status": "processing",
                "task_id": raw_document.task_id,
                "doc_id": raw_document.file_id,
                "user_id": raw_document.user_id,
                "role_id": raw_document.role_id,
                "source_uri": raw_document.source_uri,
                "started_at": started_at,
                "mode": mode,
            },
        )
        await redis.expire(status_key, settings.redis_ingest_status_ttl_seconds)
        emit_runtime_trace(
            self.logger,
            "knowledge_ingest_status_processing_set",
            status_key=status_key,
            ttl_seconds=settings.redis_ingest_status_ttl_seconds,
        )

        async with session_factory() as db_session:
            knowledge_file_service = KnowledgeFileService()
            await knowledge_file_service.update_status(
                db_session,
                file_id=raw_document.file_id,
                status="processing",
                storage_path=raw_document.source_uri,
            )
        emit_runtime_trace(
            self.logger,
            "knowledge_ingest_mysql_processing_set",
            file_id=raw_document.file_id,
        )

        try:
            emit_runtime_trace(self.logger, "knowledge_ingest_loader_started", task_id=raw_document.task_id)
            parsed = await self.loader.load(raw_document)
            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_loader_finished",
                parser_name=parsed.parser_name,
                text_chars=len(parsed.plain_text),
                text_preview=preview_text(parsed.plain_text, 160),
                section_count=len(parsed.sections),
            )

            emit_runtime_trace(self.logger, "knowledge_ingest_cleaner_started", doc_id=parsed.doc_id)
            cleaned = self.cleaner.clean(parsed)
            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_cleaner_finished",
                clean_chars=len(cleaned.clean_text),
                removed_items=cleaned.removed_items,
                clean_preview=preview_text(cleaned.clean_text, 160),
            )

            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_chunker_started",
                doc_id=cleaned.doc_id,
                role_category=role_category,
            )
            chunked = self.chunker.split(cleaned, role_category=role_category)
            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_chunker_finished",
                chunk_count=len(chunked),
                first_chunk_preview=preview_text(chunked[0].text, 160) if chunked else "",
            )

            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_embedder_started",
                chunk_count=len(chunked),
            )
            embedded = await self.embedder.embed(chunked)
            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_embedder_finished",
                embedded_count=len(embedded),
                vector_dim=len(embedded[0].embedding) if embedded else 0,
                vector_preview=embedded[0].embedding[:8] if embedded else [],
            )

            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_upload_artifact_started",
                doc_id=raw_document.file_id,
            )
            parsed_artifact_uri = await self._upload_parsed_artifact(
                raw_document=raw_document,
                parser_name=parsed.parser_name,
                title=parsed.title,
                metadata=parsed.metadata,
                removed_items=cleaned.removed_items,
                chunk_count=len(embedded),
            )
            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_upload_artifact_finished",
                parsed_artifact_uri=parsed_artifact_uri,
            )

            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_milvus_write_started",
                chunk_count=len(embedded),
                collection_name=collection_name or self._default_collection_name(),
            )
            inserted = await self._write_to_milvus(
                embedded,
                doc_id=raw_document.file_id,
                tenant_key=f"{raw_document.user_id}:{raw_document.role_id}",
                mode=mode,
                collection_name=collection_name,
                replace_doc_id=replace_doc_id,
            )
            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_milvus_write_finished",
                inserted_count=inserted,
            )

            finished_at = int(time.time())
            status = {
                "status": "success",
                "task_id": raw_document.task_id,
                "doc_id": raw_document.file_id,
                "user_id": raw_document.user_id,
                "role_id": raw_document.role_id,
                "chunk_count": inserted,
                "source_uri": raw_document.source_uri,
                "parsed_artifact_uri": parsed_artifact_uri,
                "started_at": started_at,
                "finished_at": finished_at,
                "mode": mode,
            }

            await redis.hset(status_key, mapping=status)
            await redis.expire(status_key, settings.redis_ingest_status_ttl_seconds)
            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_status_success_set",
                status_key=status_key,
                chunk_count=inserted,
            )

            async with session_factory() as db_session:
                knowledge_file_service = KnowledgeFileService()
                audit_service = AuditService()
                await knowledge_file_service.update_status(
                    db_session,
                    file_id=raw_document.file_id,
                    status="success",
                )
                if replace_doc_id:
                    await knowledge_file_service.mark_replaced(
                        db_session,
                        file_id=replace_doc_id,
                        replaced_by_file_id=raw_document.file_id,
                    )
                await audit_service.write_log(
                    db_session,
                    user_id=raw_document.user_id,
                    role_id=raw_document.role_id,
                    action="knowledge_ingest_success",
                    resource_type="knowledge_file",
                    resource_id=raw_document.file_id,
                    status="success",
                    message=f"Knowledge ingest completed with {inserted} chunks.",
                )

            logger.info(
                "document_ingested",
                task_id=raw_document.task_id,
                doc_id=raw_document.file_id,
                chunk_count=inserted,
                mode=mode,
            )
            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_completed",
                task_id=raw_document.task_id,
                doc_id=raw_document.file_id,
                chunk_count=inserted,
                parsed_artifact_uri=parsed_artifact_uri,
            )
            return status
        except Exception as exc:
            finished_at = int(time.time())
            await redis.hset(
                status_key,
                mapping={
                    "status": "failed",
                    "task_id": raw_document.task_id,
                    "doc_id": raw_document.file_id,
                    "user_id": raw_document.user_id,
                    "role_id": raw_document.role_id,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "error_message": str(exc),
                    "mode": mode,
                },
            )
            await redis.expire(status_key, settings.redis_ingest_status_ttl_seconds)

            async with session_factory() as db_session:
                knowledge_file_service = KnowledgeFileService()
                audit_service = AuditService()
                await knowledge_file_service.update_status(
                    db_session,
                    file_id=raw_document.file_id,
                    status="failed",
                )
                await audit_service.write_log(
                    db_session,
                    user_id=raw_document.user_id,
                    role_id=raw_document.role_id,
                    action="knowledge_ingest_failed",
                    resource_type="knowledge_file",
                    resource_id=raw_document.file_id,
                    status="failed",
                    message=str(exc),
                )

            emit_runtime_trace(
                self.logger,
                "knowledge_ingest_failed_detail",
                task_id=raw_document.task_id,
                doc_id=raw_document.file_id,
                error=str(exc),
            )
            logger.exception(
                "document_ingest_failed",
                task_id=raw_document.task_id,
                doc_id=raw_document.file_id,
                error=str(exc),
            )
            raise

    async def _upload_parsed_artifact(
        self,
        *,
        raw_document: RawDocument,
        parser_name: str,
        title: str,
        metadata: dict[str, Any],
        removed_items: list[str],
        chunk_count: int,
    ) -> str:
        object_name = self.storage_service.build_parsed_object_name(
            user_id=raw_document.user_id,
            role_id=raw_document.role_id,
            task_id=raw_document.task_id,
        )
        payload = {
            "task_id": raw_document.task_id,
            "doc_id": raw_document.file_id,
            "user_id": raw_document.user_id,
            "role_id": raw_document.role_id,
            "file_name": raw_document.file_name,
            "source_uri": raw_document.source_uri,
            "parser_name": parser_name,
            "title": title,
            "chunk_count": chunk_count,
            "removed_items": removed_items,
            "metadata": metadata,
        }
        emit_runtime_trace(
            self.logger,
            "knowledge_ingest_artifact_payload_ready",
            object_name=object_name,
            chunk_count=chunk_count,
            removed_items=removed_items,
            metadata_keys=list(metadata.keys())[:10],
        )
        return await self.storage_service.upload_bytes(
            bucket=get_settings().minio_bucket_parsed,
            object_name=object_name,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
            metadata={"task_id": raw_document.task_id, "doc_id": raw_document.file_id},
        )

    @log_timed("milvus_write", emit_start=False)
    async def _write_to_milvus(
        self,
        chunks: list[EmbeddedChunk],
        *,
        doc_id: str,
        tenant_key: str,
        mode: str,
        collection_name: str | None,
        replace_doc_id: str | None = None,
    ) -> int:
        if not chunks:
            emit_runtime_trace(self.logger, "milvus_write_skipped", reason="empty_chunks")
            return 0

        payload = [self._serialize_chunk(chunk) for chunk in chunks]
        collection = collection_name or self._default_collection_name()
        emit_runtime_trace(
            self.logger,
            "milvus_write_payload_ready",
            collection_name=collection,
            payload_count=len(payload),
            first_text_preview=preview_text(payload[0]["text"], 120) if payload else "",
            first_vector_preview=payload[0]["embedding"][:8] if payload else [],
        )

        if mode == "full":
            emit_runtime_trace(
                self.logger,
                "milvus_write_delete_started",
                mode=mode,
                filter_expr=f'tenant_key == "{tenant_key}"',
            )
            await asyncio.to_thread(
                self.milvus_client.delete,
                collection_name=collection,
                filter=f'tenant_key == "{tenant_key}"',
            )
        else:
            delete_doc_id = replace_doc_id or doc_id
            emit_runtime_trace(
                self.logger,
                "milvus_write_delete_started",
                mode=mode,
                filter_expr=f'doc_id == "{delete_doc_id}"',
            )
            await asyncio.to_thread(
                self.milvus_client.delete,
                collection_name=collection,
                filter=f'doc_id == "{delete_doc_id}"',
            )

        upsert_method = getattr(self.milvus_client, "upsert", None)
        if callable(upsert_method):
            emit_runtime_trace(self.logger, "milvus_write_upsert_started", collection_name=collection)
            await asyncio.to_thread(upsert_method, collection_name=collection, data=payload)
        else:
            emit_runtime_trace(self.logger, "milvus_write_insert_started", collection_name=collection)
            await asyncio.to_thread(self.milvus_client.insert, collection_name=collection, data=payload)

        emit_runtime_trace(
            self.logger,
            "milvus_write_completed",
            collection_name=collection,
            payload_count=len(payload),
        )
        return len(payload)

    def _serialize_chunk(self, chunk: EmbeddedChunk) -> dict[str, Any]:
        now = int(time.time())
        source_value = str(
            chunk.metadata.get("origin_url")
            or chunk.metadata.get("source_url")
            or chunk.source
        )
        return {
            "id": chunk.id,
            "tenant_key": chunk.tenant_key,
            "role_category": chunk.role_category,
            "text": chunk.text,
            "embedding": chunk.embedding,
            "source": source_value,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "created_at": int(chunk.metadata.get("created_at", now)),
            "updated_at": now,
        }

    def _default_collection_name(self) -> str:
        return get_settings().milvus_collection_name
