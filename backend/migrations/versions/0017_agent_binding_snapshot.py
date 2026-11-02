"""Persist the application-owned image reviewer binding."""

from alembic import op
import sqlalchemy as sa

revision = "0017_agent_binding_snapshot"
down_revision = "0016_specialist_agents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "subtasks",
        sa.Column("review_agent_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "subtasks",
        sa.Column("review_agent_digest", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("subtasks", "review_agent_digest")
    op.drop_column("subtasks", "review_agent_id")
