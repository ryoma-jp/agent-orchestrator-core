"""Align pending tool-budget approvals with task snapshots."""

from alembic import op
import sqlalchemy as sa

revision = "0009_tool_budget_snapshot"
down_revision = "0008_unbounded_tool_budget"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE tasks "
            "SET pending_tool_approval = jsonb_set("
            "pending_tool_approval, '{additional_calls}', "
            "to_jsonb(COALESCE("
            "NULLIF(config_snapshot->>'max_tool_calls_per_subtask', '')::"
            "integer, "
            "20)), false) "
            "WHERE pending_tool_approval IS NOT NULL "
            "AND COALESCE("
            "NULLIF(config_snapshot->>'max_tool_calls_per_subtask', '')::"
            "integer, "
            "20) > 0"
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE tasks "
            "SET pending_tool_approval = jsonb_set("
            "pending_tool_approval, '{additional_calls}', '20'::jsonb) "
            "WHERE pending_tool_approval IS NOT NULL"
        )
    )