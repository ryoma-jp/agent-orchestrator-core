"""Persist subtask tool budgets across SDK executions."""

from alembic import op
import sqlalchemy as sa

revision = "0004_agent_sdk_limits"
down_revision = "0003_tavily_key_configured"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "subtasks",
        sa.Column(
            "tool_calls_used",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.create_check_constraint(
        "ck_subtasks_tool_calls_used",
        "subtasks",
        "tool_calls_used >= 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_subtasks_tool_calls_used", "subtasks", type_="check"
    )
    op.drop_column("subtasks", "tool_calls_used")
