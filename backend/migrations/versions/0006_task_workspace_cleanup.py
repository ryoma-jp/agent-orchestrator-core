"""Persist task deletion requests until workspace cleanup completes."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006_task_workspace_cleanup"
down_revision = "0005_task_files"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column("deletion_requested_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "task_deletion_jobs",
        sa.Column(
            "task_id", postgresql.UUID(as_uuid=True), primary_key=True
        ),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column(
            "lease_expires_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'leased')",
            name="ck_task_deletion_jobs_status",
        ),
    )
    op.create_index(
        "ix_task_deletion_jobs_available",
        "task_deletion_jobs",
        ["status", "available_at", "lease_expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_task_deletion_jobs_available", table_name="task_deletion_jobs"
    )
    op.drop_table("task_deletion_jobs")
    op.drop_column("tasks", "deletion_requested_at")
