from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from pathlib import Path
from urllib.parse import quote

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.milvus_client import close_milvus, init_milvus
from app.db.mysql_client import close_mysql, get_mysql_session_factory, init_mysql
from app.db.redis_client import close_redis, init_redis
from app.knowledge.ingest import KnowledgeIngestService
from app.knowledge.loader import build_raw_document
from app.services.knowledge_file_service import KnowledgeFileService
from app.services.role_service import RoleService
from app.storage.minio_service import MinioStorageService

setup_logging()

COLLECTED_DIR = ROOT_DIR / "data" / "collected"
DOCTOR_ARTICLE_URL_PATTERN = re.compile(r"/(?:\d{6}|c\d{6,})/.*\.shtml$")


def _ascii_safe_metadata_value(value: str) -> str:
    if not value:
        return value
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        return quote(value, safe=":/?&=%._-")


def _is_ingestable(role_id: str, payload: dict[str, str]) -> bool:
    source_url = str(payload.get("source_url") or "")
    content = str(payload.get("content") or "")
    if role_id == "lawyer_01":
        return bool(source_url and len(content) >= 200)
    if role_id == "doctor_01":
        return bool(source_url and DOCTOR_ARTICLE_URL_PATTERN.search(source_url) and len(content) >= 200)
    if role_id == "history_01":
        return bool(source_url and len(content) >= 80)
    return True


async def ingest_role_corpus(role_id: str, max_docs: int | None = None, mode: str = "incremental") -> int:
    settings = get_settings()
    shared_user_id = settings.shared_preset_user_id
    input_dir = COLLECTED_DIR / role_id
    if not input_dir.exists():
        raise FileNotFoundError(f"Collected corpus directory not found: {input_dir}")

    await init_mysql()
    await init_redis()
    await init_milvus()

    session_factory = get_mysql_session_factory()
    storage_service = MinioStorageService()
    role_service = RoleService()
    knowledge_file_service = KnowledgeFileService()
    ingest_service = KnowledgeIngestService(storage_service=storage_service)

    count = 0
    try:
        async with session_factory() as db_session:
            role = await role_service.resolve_role(db_session, user_id=shared_user_id, role_id=role_id)

        full_reset_pending = mode == "full"
        for text_path in sorted(input_dir.glob("*.txt")):
            if max_docs is not None and count >= max_docs:
                break

            meta_path = text_path.with_suffix(".meta.json")
            if not meta_path.exists():
                continue

            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if not _is_ingestable(role_id, payload):
                continue
            file_id = str(payload.get("doc_id") or text_path.stem)
            task_id = uuid.uuid4().hex
            object_name = storage_service.build_raw_object_name(
                user_id=shared_user_id,
                role_id=role_id,
                task_id=task_id,
                file_name=text_path.name,
            )
            source_uri = await storage_service.upload_file(
                bucket=settings.minio_bucket_raw,
                object_name=object_name,
                local_path=str(text_path),
                content_type="text/plain; charset=utf-8",
                metadata={
                    "task_id": task_id,
                    "user_id": shared_user_id,
                    "role_id": role_id,
                    "source_url": _ascii_safe_metadata_value(str(payload.get("source_url") or "")),
                },
            )

            raw_document = build_raw_document(
                user_id=shared_user_id,
                role_id=role_id,
                file_name=text_path.name,
                content_type="text/plain; charset=utf-8",
                local_path=str(text_path),
                file_id=file_id,
                task_id=task_id,
                source_uri=source_uri,
                source_type="collector",
                metadata={
                    "origin_url": str(payload.get("source_url") or ""),
                    "source_url": str(payload.get("source_url") or ""),
                    "source_domain": str(payload.get("source_domain") or ""),
                    "source_name": str(payload.get("source_name") or ""),
                    "source_tier": str(payload.get("source_tier") or ""),
                    "title": str(payload.get("title") or text_path.stem),
                    "published_at": str(payload.get("published_at") or ""),
                },
            )

            effective_mode = "full" if full_reset_pending else "incremental"

            async with session_factory() as db_session:
                await knowledge_file_service.upsert_file_record(
                    db_session,
                    file_id=file_id,
                    user_id=shared_user_id,
                    role_id=role_id,
                    file_name=text_path.name,
                    content_type="text/plain; charset=utf-8",
                    storage_path=source_uri,
                    ingest_mode=effective_mode,
                    status="queued",
                )

            await ingest_service.ingest_document(
                raw_document,
                role_category=role.category,
                mode=effective_mode,
            )
            full_reset_pending = False
            count += 1
    finally:
        await close_milvus()
        await close_redis()
        await close_mysql()

    return count


async def _main_async(args: argparse.Namespace) -> None:
    count = await ingest_role_corpus(args.role_id, max_docs=args.max_docs, mode=args.mode)
    print(f"{args.role_id} ingested {count} documents into shared preset knowledge")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-id", required=True, choices=["lawyer_01", "doctor_01", "history_01"])
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--mode", choices=["incremental", "full"], default="incremental")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
