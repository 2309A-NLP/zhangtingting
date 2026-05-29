"""add knowledge file dedupe metadata

Revision ID: 20260516_0003
Revises: 20260428_0002
Create Date: 2026-05-16 13:00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260516_0003"
down_revision = "20260428_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("knowledge_files", sa.Column("file_hash", sa.String(length=64), nullable=True))
    op.add_column("knowledge_files", sa.Column("replaced_by_file_id", sa.String(length=64), nullable=True))
    op.add_column("knowledge_files", sa.Column("replaced_at", sa.DateTime(), nullable=True))
    op.execute("UPDATE knowledge_files SET file_hash = SHA2(CONCAT(user_id, ':', role_id, ':', id), 256) WHERE file_hash IS NULL")
    op.alter_column("knowledge_files", "file_hash", existing_type=sa.String(length=64), nullable=False)
    op.create_index("idx_user_role_hash", "knowledge_files", ["user_id", "role_id", "file_hash"])


def downgrade() -> None:
    op.drop_index("idx_user_role_hash", table_name="knowledge_files")
    op.drop_column("knowledge_files", "replaced_at")
    op.drop_column("knowledge_files", "replaced_by_file_id")
    op.drop_column("knowledge_files", "file_hash")
