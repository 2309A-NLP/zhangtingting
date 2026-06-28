"""add reminder delivery task

Revision ID: 20260617_0013
Revises: 20260617_0012
Create Date: 2026-06-17 13:40:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260617_0013"
down_revision: str | None = "20260617_0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("reminder_delivery_task"):
        op.create_table(
            "reminder_delivery_task",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("schedule_id", sa.Integer(), nullable=False),
            sa.Column("reminder_log_id", sa.Integer(), nullable=False),
            sa.Column("task_type", sa.String(length=32), server_default="send", nullable=False),
            sa.Column("status", sa.String(length=32), server_default="queued", nullable=False),
            sa.Column("available_at", sa.DateTime(timezone=False), nullable=False),
            sa.Column("locked_by", sa.String(length=100), nullable=True),
            sa.Column("locked_at", sa.DateTime(timezone=False), nullable=True),
            sa.Column("last_error_message", sa.String(length=500), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(["schedule_id"], ["schedule.id"]),
            sa.ForeignKeyConstraint(["reminder_log_id"], ["reminder_log.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("reminder_delivery_task")}
    if op.f("ix_reminder_delivery_task_schedule_id") not in indexes:
        op.create_index(op.f("ix_reminder_delivery_task_schedule_id"), "reminder_delivery_task", ["schedule_id"], unique=False)
    if op.f("ix_reminder_delivery_task_reminder_log_id") not in indexes:
        op.create_index(
            op.f("ix_reminder_delivery_task_reminder_log_id"),
            "reminder_delivery_task",
            ["reminder_log_id"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("reminder_delivery_task"):
        indexes = {index["name"] for index in inspector.get_indexes("reminder_delivery_task")}
        if op.f("ix_reminder_delivery_task_reminder_log_id") in indexes:
            op.drop_index(op.f("ix_reminder_delivery_task_reminder_log_id"), table_name="reminder_delivery_task")
        if op.f("ix_reminder_delivery_task_schedule_id") in indexes:
            op.drop_index(op.f("ix_reminder_delivery_task_schedule_id"), table_name="reminder_delivery_task")
        op.drop_table("reminder_delivery_task")
