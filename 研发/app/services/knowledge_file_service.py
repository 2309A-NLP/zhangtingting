from __future__ import annotations
'''知识库文件的数据库操作服务，负责管理 knowledge_files 表的增删改查。'''
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger

logger = get_logger(__name__)


class KnowledgeFileService:
    async def find_active_duplicate(
        self,
        db_session: AsyncSession,
        *,
        user_id: str,
        role_id: str,
        file_hash: str,
    ) -> dict[str, str] | None:
        stmt = text(
            """
            SELECT id, file_name, status, storage_path, ingest_mode
            FROM knowledge_files
            WHERE user_id = :user_id
              AND role_id = :role_id
              AND file_hash = :file_hash
              AND replaced_by_file_id IS NULL
              AND status IN ('queued', 'processing', 'success', 'failed')
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        result = await db_session.execute(
            stmt,
            {"user_id": user_id, "role_id": role_id, "file_hash": file_hash},
        )
        row = result.mappings().first()
        return dict(row) if row else None

    # 插入新记录，如果已存在则更新
    async def upsert_file_record(
        self,
        db_session: AsyncSession,
        *,
        file_id: str,
        user_id: str,
        role_id: str,
        file_name: str,
        file_hash: str,
        content_type: str,
        storage_path: str,
        ingest_mode: str,
        status: str,
    ) -> None:
        stmt = text(
            """
            INSERT INTO knowledge_files (id, user_id, role_id, file_name, file_hash, content_type, storage_path, ingest_mode, status, created_at, updated_at)
            VALUES (:id, :user_id, :role_id, :file_name, :file_hash, :content_type, :storage_path, :ingest_mode, :status, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
              file_name = VALUES(file_name),
              file_hash = VALUES(file_hash),
              content_type = VALUES(content_type),
              storage_path = VALUES(storage_path),
              ingest_mode = VALUES(ingest_mode),
              status = VALUES(status),
              updated_at = NOW()
            """
        )
        await db_session.execute(
            stmt,
            {
                "id": file_id,
                "user_id": user_id,
                "role_id": role_id,
                "file_name": file_name,
                "file_hash": file_hash,
                "content_type": content_type,
                "storage_path": storage_path, # 存储路径（MinIO URI）
                "ingest_mode": ingest_mode, # 摄入模式
                "status": status,
            },
        )
        await db_session.commit()
        logger.info("knowledge_file_upserted", file_id=file_id, user_id=user_id, role_id=role_id, status=status)

    async def mark_replaced(
        self,
        db_session: AsyncSession,
        *,
        file_id: str,
        replaced_by_file_id: str,
    ) -> None:
        stmt = text(
            """
            UPDATE knowledge_files
            SET status = 'replaced',
                replaced_by_file_id = :replaced_by_file_id,
                replaced_at = NOW(),
                updated_at = NOW()
            WHERE id = :id
            """
        )
        await db_session.execute(
            stmt,
            {"id": file_id, "replaced_by_file_id": replaced_by_file_id},
        )
        await db_session.commit()
        logger.info(
            "knowledge_file_marked_replaced",
            file_id=file_id,
            replaced_by_file_id=replaced_by_file_id,
        )

    # 更新文件的处理状态，可选同时更新存储路径
    async def update_status(
        self,
        db_session: AsyncSession,
        *,
        file_id: str,
        status: str,
        storage_path: str | None = None,
    ) -> None:
        if storage_path:
            stmt = text(
                """
                UPDATE knowledge_files
                SET status = :status, storage_path = :storage_path, updated_at = NOW()
                WHERE id = :id
                """
            )
            params = {"id": file_id, "status": status, "storage_path": storage_path}
        else:
            stmt = text(
                """
                UPDATE knowledge_files
                SET status = :status, updated_at = NOW()
                WHERE id = :id
                """
            )
            params = {"id": file_id, "status": status}
        await db_session.execute(stmt, params)
        await db_session.commit()
        logger.info("knowledge_file_status_updated", file_id=file_id, status=status)
