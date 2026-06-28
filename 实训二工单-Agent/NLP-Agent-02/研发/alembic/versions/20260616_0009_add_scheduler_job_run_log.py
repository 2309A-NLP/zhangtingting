"""add scheduler job run log

Revision ID: 20260616_0009
Revises: 20260616_0008
Create Date: 2026-06-16 13:20:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260616_0009"
down_revision: str | None = "20260616_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("scheduler_job_run_log"):
        op.create_table(
            "scheduler_job_run_log",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("job_id", sa.String(length=100), nullable=False),
            sa.Column("job_name", sa.String(length=100), nullable=False),
            sa.Column("trigger_name", sa.String(length=64), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=False), nullable=False),
            sa.Column("finished_at", sa.DateTime(timezone=False), nullable=True),
            sa.Column("status", sa.String(length=32), server_default="running", nullable=False),
            sa.Column("processed_count", sa.Integer(), server_default="0", nullable=False),
            sa.Column("error_message", sa.String(length=500), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=False),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    existing_indexes = {index["name"] for index in sa.inspect(bind).get_indexes("scheduler_job_run_log")}
    if op.f("ix_scheduler_job_run_log_job_id") not in existing_indexes:
        op.create_index(op.f("ix_scheduler_job_run_log_job_id"), "scheduler_job_run_log", ["job_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if inspector.has_table("scheduler_job_run_log"):
        existing_indexes = {index["name"] for index in inspector.get_indexes("scheduler_job_run_log")}
        if op.f("ix_scheduler_job_run_log_job_id") in existing_indexes:
            op.drop_index(op.f("ix_scheduler_job_run_log_job_id"), table_name="scheduler_job_run_log")
        op.drop_table("scheduler_job_run_log")
