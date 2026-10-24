"""Track whether the task title was generated from its initial instruction."""

from alembic import op
import sqlalchemy as sa

revision = "0011_task_title_generated"
down_revision = "0010_task_title"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column(
            "title_generated",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("tasks", "title_generated")
