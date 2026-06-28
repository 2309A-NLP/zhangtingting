"""add admin access audit log

Revision ID: 20260616_0008
Revises: 20260616_0007
Create Date: 2026-06-16 21:20:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260616_0008"
down_revision: str | None = "20260616_0007"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("admin_access_audit_log"):
        op.create_table(
            "admin_access_audit_log",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("path", sa.String(length=255), nullable=False),
            sa.Column("method", sa.String(length=16), nullable=False),
            sa.Column("client_host", sa.String(length=64), nullable=True),
            sa.Column("request_id", sa.String(length=64), nullable=True),
            sa.Column("access_granted", sa.Boolean(), server_default=sa.text("0"), nullable=False),
            sa.Column("auth_mode", sa.String(length=32), nullable=False),
            sa.Column("failure_reason", sa.String(length=255), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=False),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=False),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    existing_indexes = {index["name"] for index in sa.inspect(bind).get_indexes("admin_access_audit_log")}
    if op.f("ix_admin_access_audit_log_path") not in existing_indexes:
        op.create_index(op.f("ix_admin_access_audit_log_path"), "admin_access_audit_log", ["path"], unique=False)
    if op.f("ix_admin_access_audit_log_request_id") not in existing_indexes:
        op.create_index(
            op.f("ix_admin_access_audit_log_request_id"),
            "admin_access_audit_log",
            ["request_id"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if inspector.has_table("admin_access_audit_log"):
        existing_indexes = {index["name"] for index in inspector.get_indexes("admin_access_audit_log")}
        if op.f("ix_admin_access_audit_log_request_id") in existing_indexes:
            op.drop_index(op.f("ix_admin_access_audit_log_request_id"), table_name="admin_access_audit_log")
        if op.f("ix_admin_access_audit_log_path") in existing_indexes:
            op.drop_index(op.f("ix_admin_access_audit_log_path"), table_name="admin_access_audit_log")
        op.drop_table("admin_access_audit_log")
