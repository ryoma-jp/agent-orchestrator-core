"""Persist pending tool-budget approvals and approved subtask limits."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0007_tool_budget_approval"
down_revision = "0006_task_workspace_cleanup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column("pending_tool_approval", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "subtasks",
        sa.Column("tool_calls_limit", sa.Integer(), nullable=True),
    )
    op.create_check_constraint(
        "ck_subtasks_tool_calls_limit",
        "subtasks",
        "tool_calls_limit IS NULL OR tool_calls_limit BETWEEN 0 AND 100",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_subtasks_tool_calls_limit", "subtasks", type_="check"
    )
    op.drop_column("subtasks", "tool_calls_limit")
    op.drop_column("tasks", "pending_tool_approval")