"""add agent conversation state table

Revision ID: 20260615_0003
Revises: 20260615_0002
Create Date: 2026-06-15 21:20:00
"""

import sqlalchemy as sa
from alembic import op

revision = "20260615_0003"
down_revision = "20260615_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_conversation_state",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False),
        sa.Column("intent", sa.String(length=32), nullable=False),
        sa.Column("agent_state", sa.String(length=32), nullable=False),
        sa.Column("tool_name", sa.String(length=64), nullable=True),
        sa.Column("tool_arguments_json", sa.Text(), nullable=True),
        sa.Column("user_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_agent_conversation_state_session_id",
        "agent_conversation_state",
        ["session_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_agent_conversation_state_session_id", table_name="agent_conversation_state")
    op.drop_table("agent_conversation_state")
