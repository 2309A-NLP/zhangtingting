"""add suggested inputs to conversation history

Revision ID: 20260616_0007
Revises: 20260616_0006
Create Date: 2026-06-16 16:20:00
"""

import sqlalchemy as sa
from alembic import op

revision = "20260616_0007"
down_revision = "20260616_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_conversation_history",
        sa.Column("suggested_inputs_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_conversation_history", "suggested_inputs_json")
