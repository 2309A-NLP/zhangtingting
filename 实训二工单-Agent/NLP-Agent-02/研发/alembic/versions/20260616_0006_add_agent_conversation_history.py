"""add agent conversation history table

Revision ID: 20260616_0006
Revises: 20260616_0005
Create Date: 2026-06-16 15:05:00
"""

import sqlalchemy as sa
from alembic import op

revision = "20260616_0006"
down_revision = "20260616_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_conversation_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False),
        sa.Column("user_input", sa.Text(), nullable=False),
        sa.Column("confirmed", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.Column("context_json", sa.Text(), nullable=True),
        sa.Column("parser_source", sa.String(length=32), nullable=True),
        sa.Column("intent", sa.String(length=32), nullable=False),
        sa.Column("agent_state", sa.String(length=32), nullable=False),
        sa.Column("tool_name", sa.String(length=64), nullable=True),
        sa.Column("tool_arguments_json", sa.Text(), nullable=True),
        sa.Column("missing_fields_json", sa.Text(), nullable=True),
        sa.Column("target_id", sa.Integer(), nullable=True),
        sa.Column("execution_result_json", sa.Text(), nullable=True),
        sa.Column("user_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_agent_conversation_history_session_id",
        "agent_conversation_history",
        ["session_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_agent_conversation_history_session_id", table_name="agent_conversation_history")
    op.drop_table("agent_conversation_history")
