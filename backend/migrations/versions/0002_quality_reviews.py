"""Persist versioned task plans and quality reviews."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_quality_reviews"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tasks", sa.Column("completion_report", postgresql.JSONB())
    )
    op.drop_constraint("ck_tasks_status", "tasks", type_="check")
    op.create_check_constraint(
        "ck_tasks_status",
        "tasks",
        "status IN ('queued', 'planning', 'running', 'waiting_approval', "
        "'succeeded', 'partially_completed', 'failed', 'cancelled', "
        "'needs_attention')",
    )

    op.add_column(
        "subtasks",
        sa.Column("artifact_goal", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "subtasks",
        sa.Column("plan_position", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "subtasks",
        sa.Column(
            "acceptance_checklist",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "subtasks", sa.Column("plan_version", sa.Integer(), nullable=True)
    )
    op.add_column(
        "subtasks",
        sa.Column(
            "quality_revision_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.drop_constraint("ck_subtasks_status", "subtasks", type_="check")
    op.create_check_constraint(
        "ck_subtasks_status",
        "subtasks",
        "status IN ('pending', 'ready', 'running', 'waiting_approval', "
        "'reviewing', 'revising', 'accepted', 'partially_accepted', "
        "'succeeded', 'failed', 'blocked', 'cancelled')",
    )

    op.create_table(
        "task_plans",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("conversation_sequence", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("final_checklist", postgresql.JSONB(), nullable=False),
        sa.Column("change_reason", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("task_id", "version", name="uq_task_plans_version"),
    )
    op.create_index("ix_task_plans_task_id", "task_plans", ["task_id"])

    op.create_table(
        "quality_reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("plan_version", sa.Integer(), nullable=False),
        sa.Column("target_type", sa.String(16), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True)),
        sa.Column("revision_round", sa.Integer(), nullable=False),
        sa.Column("checklist_results", postgresql.JSONB(), nullable=False),
        sa.Column("verdict", sa.String(32), nullable=False),
        sa.Column("reviewer_model_id", sa.String(200), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "target_type IN ('subtask', 'final')",
            name="ck_quality_reviews_target_type",
        ),
        sa.CheckConstraint(
            "verdict IN ('accepted', 'revision_requested', "
            "'partially_accepted')",
            name="ck_quality_reviews_verdict",
        ),
    )
    op.create_index("ix_quality_reviews_task_id", "quality_reviews", ["task_id"])
    op.create_index(
        "ix_quality_reviews_task_plan",
        "quality_reviews",
        ["task_id", "plan_version"],
    )


def downgrade() -> None:
    op.drop_index("ix_quality_reviews_task_plan", table_name="quality_reviews")
    op.drop_index("ix_quality_reviews_task_id", table_name="quality_reviews")
    op.drop_table("quality_reviews")
    op.drop_index("ix_task_plans_task_id", table_name="task_plans")
    op.drop_table("task_plans")

    op.execute(
        "UPDATE subtasks SET status = 'succeeded' "
        "WHERE status IN ('reviewing', 'revising', 'accepted', "
        "'partially_accepted')"
    )
    op.drop_constraint("ck_subtasks_status", "subtasks", type_="check")
    op.create_check_constraint(
        "ck_subtasks_status",
        "subtasks",
        "status IN ('pending', 'ready', 'running', 'waiting_approval', "
        "'succeeded', 'failed', 'blocked', 'cancelled')",
    )
    op.drop_column("subtasks", "quality_revision_count")
    op.drop_column("subtasks", "plan_version")
    op.drop_column("subtasks", "acceptance_checklist")
    op.drop_column("subtasks", "artifact_goal")
    op.drop_column("subtasks", "plan_position")

    op.execute(
        "UPDATE tasks SET status = 'failed' "
        "WHERE status = 'partially_completed'"
    )
    op.drop_constraint("ck_tasks_status", "tasks", type_="check")
    op.create_check_constraint(
        "ck_tasks_status",
        "tasks",
        "status IN ('queued', 'planning', 'running', 'waiting_approval', "
        "'succeeded', 'failed', 'cancelled', 'needs_attention')",
    )
    op.drop_column("tasks", "completion_report")