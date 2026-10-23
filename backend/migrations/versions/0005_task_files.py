"""Persist metadata for files stored in task workspaces."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_task_files"
down_revision = "0004_agent_sdk_limits"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "task_files",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("workspace_rel_path", sa.Text(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("source_kind", sa.String(length=32), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role IN ('input', 'source', 'intermediate', 'final')",
            name="ck_task_files_role",
        ),
        sa.UniqueConstraint(
            "task_id", "workspace_rel_path", name="uq_task_files_path"
        ),
    )
    op.create_index(
        "ix_task_files_task_path",
        "task_files",
        ["task_id", "workspace_rel_path"],
    )
    op.create_index("ix_task_files_task_id", "task_files", ["task_id"])


def downgrade() -> None:
    op.drop_index("ix_task_files_task_id", table_name="task_files")
    op.drop_index("ix_task_files_task_path", table_name="task_files")
    op.drop_table("task_files")
