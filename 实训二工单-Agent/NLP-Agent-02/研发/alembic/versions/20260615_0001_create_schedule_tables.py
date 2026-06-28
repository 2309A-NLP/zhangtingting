"""create schedule and reminder tables

Revision ID: 20260615_0001
Revises: None
Create Date: 2026-06-15 20:10:00
"""

import sqlalchemy as sa
from alembic import op

revision = "20260615_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "schedule",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("content", sa.String(length=255), nullable=False),
        sa.Column("schedule_date", sa.Date(), nullable=True),
        sa.Column("schedule_time", sa.Time(), nullable=False),
        sa.Column("cycle_rule", sa.String(length=32), server_default="once", nullable=False),
        sa.Column("cycle_value", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("source_text", sa.String(length=500), nullable=True),
        sa.Column("next_trigger_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "reminder_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("schedule_id", sa.Integer(), nullable=False),
        sa.Column("planned_trigger_at", sa.DateTime(), nullable=False),
        sa.Column("reminded_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["schedule_id"], ["schedule.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_reminder_log_schedule_id", "reminder_log", ["schedule_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_reminder_log_schedule_id", table_name="reminder_log")
    op.drop_table("reminder_log")
    op.drop_table("schedule")
