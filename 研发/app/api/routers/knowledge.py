from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id, get_db_session, get_role_service, require_user_match
from app.api.schemas import Envelope, KnowledgeTaskStatusResponse, KnowledgeUploadResponse
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.request_context import get_request_id
from app.core.response import success_response
from app.db.mysql_client import get_mysql_session_factory
from app.db.redis_client import get_redis, ingest_status_key
from app.knowledge.loader import build_raw_document
from app.knowledge.task_queue import KnowledgeIngestTask, get_knowledge_task_queue
from app.services.audit_service import AuditService
from app.services.knowledge_file_service import KnowledgeFileService
from app.services.role_service import RoleService
from app.storage.minio_service import MinioStorageService

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
logger = get_logger(__name__)


@router.post("/upload", response_model=Envelope[KnowledgeUploadResponse])
async def upload_knowledge(
    user_id: str = Form(...),
    role_id: str = Form(...),
    mode: str = Form(default="incremental"),
    overwrite: bool = Form(default=False),
    file: UploadFile = File(...),
    db_session: AsyncSession = Depends(get_db_session),
    current_user_id: str | None = Depends(get_current_user_id),
    role_service: RoleService = Depends(get_role_service),
):
    require_user_match(user_id, current_user_id)
    settings = get_settings()
    request_id = get_request_id()
    role = await role_service.resolve_role(db_session, user_id=user_id, role_id=role_id)

    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    if suffix not in settings.allowed_upload_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    task_id = uuid.uuid4().hex
    local_path = upload_dir / f"{task_id}_{file.filename}"
    file_hash = hashlib.sha256()

    size = 0
    async with aiofiles.open(local_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            file_hash.update(chunk)
            if size > settings.max_upload_size_mb * 1024 * 1024:
                await out.close()
                local_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large.")
            await out.write(chunk)

    file_hash_hex = file_hash.hexdigest()
    knowledge_file_service = KnowledgeFileService()
    duplicate = await knowledge_file_service.find_active_duplicate(
        db_session,
        user_id=user_id,
        role_id=role_id,
        file_hash=file_hash_hex,
    )
    if duplicate and duplicate["status"] in {"queued", "processing"}:
        local_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail=(
                "A file with the same content is already being processed for this role. "
                f"existing_file_id={duplicate['id']}"
            ),
        )
    if duplicate and not overwrite:
        local_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail=(
                "Duplicate file detected for this role. "
                f"existing_file_id={duplicate['id']}. "
                "Set overwrite=true to replace the previous version."
            ),
        )

    replace_doc_id = duplicate["id"] if duplicate and overwrite else None

    storage_service = MinioStorageService()
    raw_object_name = storage_service.build_raw_object_name(
        user_id=user_id,
        role_id=role_id,
        task_id=task_id,
        file_name=file.filename or local_path.name,
    )
    source_uri = await storage_service.upload_file(
        bucket=settings.minio_bucket_raw,
        object_name=raw_object_name,
        local_path=str(local_path),
        content_type=file.content_type or "application/octet-stream",
        metadata={"task_id": task_id, "user_id": user_id, "role_id": role_id},
    )

    raw_document = build_raw_document(
        user_id=user_id,
        role_id=role_id,
        file_name=file.filename or local_path.name,
        content_type=file.content_type or "application/octet-stream",
        local_path=str(local_path),
        file_id=task_id,
        task_id=task_id,
        source_uri=source_uri,
        metadata={
            "task_id": task_id,
            "object_name": raw_object_name,
            "file_hash": file_hash_hex,
            "overwrite": overwrite,
            "replace_doc_id": replace_doc_id,
        },
    )

    queue = get_knowledge_task_queue()
    await queue.enqueue(
        KnowledgeIngestTask(
            task_id=task_id,
            user_id=user_id,
            role_id=role_id,
            role_category=role.category,
            mode="full" if mode == "full" else "incremental",
            raw_document=raw_document,
            replace_doc_id=replace_doc_id,
        )
    )

    await knowledge_file_service.upsert_file_record(
        db_session,
        file_id=task_id,
        user_id=user_id,
        role_id=role_id,
        file_name=file.filename or local_path.name,
        file_hash=file_hash_hex,
        content_type=file.content_type or "application/octet-stream",
        storage_path=source_uri,
        ingest_mode="full" if mode == "full" else "incremental",
        status="queued",
    )

    audit_service = AuditService()
    await audit_service.write_log(
        db_session,
        request_id=request_id,
        user_id=user_id,
        role_id=role_id,
        action="knowledge_upload_queued",
        resource_type="knowledge_file",
        resource_id=task_id,
        status="success",
        message=f"Knowledge upload queued for role {role.role_id}.",
    )

    logger.info(
        "knowledge_upload_queued",
        task_id=task_id,
        user_id=user_id,
        role_id=role_id,
        overwrite=overwrite,
        duplicate_of_file_id=replace_doc_id,
    )

    return success_response(
        KnowledgeUploadResponse(
            task_id=task_id,
            user_id=user_id,
            role_id=role_id,
            mode="full" if mode == "full" else "incremental",
            status="queued",
            overwrite=overwrite,
            duplicate_of_file_id=replace_doc_id,
            uploaded_at=datetime.utcnow(),
        )
    )


@router.get("/tasks/{task_id}", response_model=Envelope[KnowledgeTaskStatusResponse])
async def get_knowledge_task_status(
    task_id: str,
    user_id: str,
    role_id: str,
    current_user_id: str | None = Depends(get_current_user_id),
):
    require_user_match(user_id, current_user_id)
    request_id = get_request_id()
    redis = get_redis()
    payload = await redis.hgetall(ingest_status_key(user_id, role_id, task_id))
    if not payload:
        raise HTTPException(status_code=404, detail=f"Knowledge task not found: {task_id}")

    try:
        session_factory = get_mysql_session_factory()
        async with session_factory() as audit_session:
            audit_service = AuditService()
            await audit_service.write_log(
                audit_session,
                request_id=request_id,
                user_id=user_id,
                role_id=role_id,
                action="knowledge_task_status_read",
                resource_type="knowledge_file",
                resource_id=task_id,
                status="success",
                message=f"Knowledge task status read: {payload.get('status', 'queued')}.",
            )
    except RuntimeError:
        logger.warning("knowledge_task_status_audit_skipped", task_id=task_id)

    return success_response(
        KnowledgeTaskStatusResponse(
            task_id=payload.get("task_id", task_id),
            user_id=payload.get("user_id", user_id),
            role_id=payload.get("role_id", role_id),
            mode=payload.get("mode", "incremental"),
            status=payload.get("status", "queued"),
            doc_id=payload.get("doc_id"),
            source_uri=payload.get("source_uri"),
            parsed_artifact_uri=payload.get("parsed_artifact_uri"),
            chunk_count=int(payload["chunk_count"]) if payload.get("chunk_count") else None,
            error_message=payload.get("error_message"),
            started_at=int(payload["started_at"]) if payload.get("started_at") else None,
            finished_at=int(payload["finished_at"]) if payload.get("finished_at") else None,
        )
    )
