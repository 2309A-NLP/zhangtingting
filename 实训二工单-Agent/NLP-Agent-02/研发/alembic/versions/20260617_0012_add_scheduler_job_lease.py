"""add scheduler job lease

Revision ID: 20260617_0012
Revises: 20260617_0011
Create Date: 2026-06-17 12:20:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260617_0012"
down_revision: str | None = "20260617_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("scheduler_job_lease"):
        op.create_table(
            "scheduler_job_lease",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("job_id", sa.String(length=100), nullable=False),
            sa.Column("owner_id", sa.String(length=100), nullable=False),
            sa.Column("locked_until", sa.DateTime(timezone=False), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("job_id"),
        )
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("scheduler_job_lease")}
    if op.f("ix_scheduler_job_lease_job_id") not in indexes:
        op.create_index(op.f("ix_scheduler_job_lease_job_id"), "scheduler_job_lease", ["job_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("scheduler_job_lease"):
        indexes = {index["name"] for index in inspector.get_indexes("scheduler_job_lease")}
        if op.f("ix_scheduler_job_lease_job_id") in indexes:
            op.drop_index(op.f("ix_scheduler_job_lease_job_id"), table_name="scheduler_job_lease")
        op.drop_table("scheduler_job_lease")
