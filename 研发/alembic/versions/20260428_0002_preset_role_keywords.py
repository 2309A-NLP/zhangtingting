"""add preset role keywords

Revision ID: 20260428_0002
Revises: 20260425_0001
Create Date: 2026-04-28 20:30:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260428_0002"
down_revision = "20260425_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "preset_role_keywords",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("role_id", sa.String(length=64), nullable=False),
        sa.Column("keyword", sa.String(length=128), nullable=False),
        sa.Column("weight", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("uk_role_keyword", "preset_role_keywords", ["role_id", "keyword"], unique=True)
    op.create_index("idx_role_enabled", "preset_role_keywords", ["role_id", "is_enabled"])


def downgrade() -> None:
    op.drop_index("idx_role_enabled", table_name="preset_role_keywords")
    op.drop_index("uk_role_keyword", table_name="preset_role_keywords")
    op.drop_table("preset_role_keywords")
