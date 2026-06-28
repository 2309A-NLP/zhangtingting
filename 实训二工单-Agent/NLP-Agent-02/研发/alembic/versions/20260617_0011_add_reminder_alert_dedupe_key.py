"""add reminder alert dedupe key

Revision ID: 20260617_0011
Revises: 20260617_0010
Create Date: 2026-06-17 11:10:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260617_0011"
down_revision: str | None = "20260617_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("reminder_alert_log")}
    if "dedupe_key" not in columns:
        op.add_column("reminder_alert_log", sa.Column("dedupe_key", sa.String(length=255), nullable=True))

    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("reminder_alert_log")}
    if op.f("ix_reminder_alert_log_dedupe_key") not in indexes:
        op.create_index(op.f("ix_reminder_alert_log_dedupe_key"), "reminder_alert_log", ["dedupe_key"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("reminder_alert_log"):
        indexes = {index["name"] for index in inspector.get_indexes("reminder_alert_log")}
        if op.f("ix_reminder_alert_log_dedupe_key") in indexes:
            op.drop_index(op.f("ix_reminder_alert_log_dedupe_key"), table_name="reminder_alert_log")
        columns = {column["name"] for column in inspector.get_columns("reminder_alert_log")}
        if "dedupe_key" in columns:
            op.drop_column("reminder_alert_log", "dedupe_key")
