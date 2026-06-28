"""add llm audit log

Revision ID: 20260616_0005
Revises: 20260615_0004
Create Date: 2026-06-16 14:20:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260616_0005"
down_revision: str | None = "20260615_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_audit_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False),
        sa.Column("user_input", sa.Text(), nullable=False),
        sa.Column("parser_stage", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("model_name", sa.String(length=128), nullable=True),
        sa.Column("success", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.Column("request_payload_json", sa.Text(), nullable=True),
        sa.Column("raw_response_text", sa.Text(), nullable=True),
        sa.Column("parsed_response_json", sa.Text(), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_llm_audit_log_session_id"), "llm_audit_log", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_llm_audit_log_session_id"), table_name="llm_audit_log")
    op.drop_table("llm_audit_log")
