"""Allow user-approved tool budgets to grow without a cumulative cap."""

from alembic import op
import sqlalchemy as sa

revision = "0008_unbounded_tool_budget"
down_revision = "0007_tool_budget_approval"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_subtasks_tool_calls_limit", "subtasks", type_="check"
    )

    op.alter_column(
        "subtasks",
        "tool_calls_used",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=False,
        existing_server_default="0",
    )
    op.alter_column(
        "subtasks",
        "tool_calls_limit",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=True,
    )
    op.create_check_constraint(
        "ck_subtasks_tool_calls_limit",
        "subtasks",
        "tool_calls_limit IS NULL OR tool_calls_limit >= 0",
    )
    op.execute(
        sa.text(
            "UPDATE tasks "
            "SET pending_tool_approval = jsonb_set("
            "pending_tool_approval, '{additional_calls}', '20'::jsonb) "
            "WHERE pending_tool_approval IS NOT NULL "
            "AND pending_tool_approval->>'additional_calls' IS NOT NULL"
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            "DO $$ BEGIN "
            "IF EXISTS (SELECT 1 FROM subtasks "
            "WHERE tool_calls_limit > 100 "
            "OR tool_calls_used > 2147483647) THEN "
            "RAISE EXCEPTION 'Cannot downgrade tool budgets above 100 calls'; "
            "END IF; END $$"
        )
    )
    op.execute(
        sa.text(
            "UPDATE tasks "
            "SET pending_tool_approval = jsonb_set("
            "pending_tool_approval, '{additional_calls}', '5'::jsonb) "
            "WHERE pending_tool_approval IS NOT NULL "
            "AND pending_tool_approval->>'additional_calls' IS NOT NULL"
        )
    )
    op.drop_constraint(
        "ck_subtasks_tool_calls_limit", "subtasks", type_="check"
    )
    op.alter_column(
        "subtasks",
        "tool_calls_used",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=False,
        existing_server_default="0",
    )
    op.alter_column(
        "subtasks",
        "tool_calls_limit",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=True,
    )
    op.create_check_constraint(
        "ck_subtasks_tool_calls_limit",
        "subtasks",
        "tool_calls_limit IS NULL OR tool_calls_limit BETWEEN 0 AND 100",
    )
