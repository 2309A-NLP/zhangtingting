"""add expires_at to agent conversation state

Revision ID: 20260615_0004
Revises: 20260615_0003
Create Date: 2026-06-15 21:50:00
"""

import sqlalchemy as sa
from alembic import op

revision = "20260615_0004"
down_revision = "20260615_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_conversation_state",
        sa.Column("expires_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_conversation_state", "expires_at")
