"""Distinguish catalog reviewer selections from legacy bindings."""

from alembic import op
import sqlalchemy as sa

revision = "0018_catalog_review_binding"
down_revision = "0017_agent_binding_snapshot"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "subtasks",
        sa.Column("review_binding_mode", sa.String(length=16), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("subtasks", "review_binding_mode")
