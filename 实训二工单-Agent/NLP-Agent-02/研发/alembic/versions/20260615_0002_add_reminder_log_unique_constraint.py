"""add unique constraint for reminder log

Revision ID: 20260615_0002
Revises: 20260615_0001
Create Date: 2026-06-15 20:45:00
"""

from alembic import op

revision = "20260615_0002"
down_revision = "20260615_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("reminder_log") as batch_op:
        batch_op.create_unique_constraint(
            "uq_reminder_log_schedule_trigger",
            ["schedule_id", "planned_trigger_at"],
        )


def downgrade() -> None:
    with op.batch_alter_table("reminder_log") as batch_op:
        batch_op.drop_constraint("uq_reminder_log_schedule_trigger", type_="unique")
