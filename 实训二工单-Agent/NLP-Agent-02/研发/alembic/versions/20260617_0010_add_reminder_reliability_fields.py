"""add reminder reliability fields

Revision ID: 20260617_0010
Revises: 20260616_0009
Create Date: 2026-06-17 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260617_0010"
down_revision: str | None = "20260616_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _column_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    reminder_columns = _column_names("reminder_log")
    if "attempt_count" not in reminder_columns:
        op.add_column("reminder_log", sa.Column("attempt_count", sa.Integer(), server_default="1", nullable=False))
    if "last_attempt_at" not in reminder_columns:
        op.add_column("reminder_log", sa.Column("last_attempt_at", sa.DateTime(timezone=False), nullable=True))
    if "next_retry_at" not in reminder_columns:
        op.add_column("reminder_log", sa.Column("next_retry_at", sa.DateTime(timezone=False), nullable=True))

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("reminder_alert_log"):
        op.create_table(
            "reminder_alert_log",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("schedule_id", sa.Integer(), nullable=False),
            sa.Column("reminder_log_id", sa.Integer(), nullable=True),
            sa.Column("alert_type", sa.String(length=64), nullable=False),
            sa.Column("alert_channel", sa.String(length=32), nullable=False),
            sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
            sa.Column("message", sa.String(length=500), nullable=False),
            sa.Column("sent_at", sa.DateTime(timezone=False), nullable=True),
            sa.Column("error_message", sa.String(length=500), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(["schedule_id"], ["schedule.id"]),
            sa.ForeignKeyConstraint(["reminder_log_id"], ["reminder_log.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
    existing_indexes = {index["name"] for index in sa.inspect(bind).get_indexes("reminder_alert_log")}
    if op.f("ix_reminder_alert_log_schedule_id") not in existing_indexes:
        op.create_index(op.f("ix_reminder_alert_log_schedule_id"), "reminder_alert_log", ["schedule_id"], unique=False)
    if op.f("ix_reminder_alert_log_reminder_log_id") not in existing_indexes:
        op.create_index(
            op.f("ix_reminder_alert_log_reminder_log_id"),
            "reminder_alert_log",
            ["reminder_log_id"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("reminder_alert_log"):
        existing_indexes = {index["name"] for index in inspector.get_indexes("reminder_alert_log")}
        if op.f("ix_reminder_alert_log_reminder_log_id") in existing_indexes:
            op.drop_index(op.f("ix_reminder_alert_log_reminder_log_id"), table_name="reminder_alert_log")
        if op.f("ix_reminder_alert_log_schedule_id") in existing_indexes:
            op.drop_index(op.f("ix_reminder_alert_log_schedule_id"), table_name="reminder_alert_log")
        op.drop_table("reminder_alert_log")

    reminder_columns = _column_names("reminder_log")
    if "next_retry_at" in reminder_columns:
        op.drop_column("reminder_log", "next_retry_at")
    if "last_attempt_at" in reminder_columns:
        op.drop_column("reminder_log", "last_attempt_at")
    if "attempt_count" in reminder_columns:
        op.drop_column("reminder_log", "attempt_count")
