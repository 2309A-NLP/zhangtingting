"""initial schema

Revision ID: 20260425_0001
Revises: None
Create Date: 2026-04-25 10:35:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260425_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("username", name="idx_username"),
    )

    op.create_table(
        "preset_roles",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_category", "preset_roles", ["category"])

    op.create_table(
        "custom_roles",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False, server_default="general"),
        sa.Column("role_type", sa.String(length=16), nullable=False, server_default="custom"),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_user_id", "custom_roles", ["user_id"])
    op.create_index("idx_user_role_type", "custom_roles", ["user_id", "role_type"])
    op.create_index("uk_user_role_name", "custom_roles", ["user_id", "name"], unique=True)

    op.create_table(
        "conversations",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("role_id", sa.String(length=64), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("tokens_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_user_role_time", "conversations", ["user_id", "role_id", "timestamp"])

    op.create_table(
        "user_role_mapping",
        sa.Column("user_id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("role_id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("last_used_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("total_interactions", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("idx_user_role", "user_role_mapping", ["user_id", "role_id"])

    op.create_table(
        "knowledge_files",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("role_id", sa.String(length=64), nullable=False),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("storage_path", sa.String(length=512), nullable=False),
        sa.Column("ingest_mode", sa.String(length=16), nullable=False, server_default="incremental"),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="queued"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_user_role_status", "knowledge_files", ["user_id", "role_id", "status"])

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("user_id", sa.String(length=64), nullable=True),
        sa.Column("role_id", sa.String(length=64), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("resource_type", sa.String(length=64), nullable=True),
        sa.Column("resource_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="success"),
        sa.Column("message", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_user_action_time", "audit_logs", ["user_id", "action", "created_at"])


def downgrade() -> None:
    op.drop_index("idx_user_action_time", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index("idx_user_role_status", table_name="knowledge_files")
    op.drop_table("knowledge_files")
    op.drop_index("idx_user_role", table_name="user_role_mapping")
    op.drop_table("user_role_mapping")
    op.drop_index("idx_user_role_time", table_name="conversations")
    op.drop_table("conversations")
    op.drop_index("uk_user_role_name", table_name="custom_roles")
    op.drop_index("idx_user_role_type", table_name="custom_roles")
    op.drop_index("idx_user_id", table_name="custom_roles")
    op.drop_table("custom_roles")
    op.drop_index("idx_category", table_name="preset_roles")
    op.drop_table("preset_roles")
    op.drop_table("users")
