"""Store manager-generated task titles."""

from alembic import op
import sqlalchemy as sa

revision = "0010_task_title"
down_revision = "0009_tool_budget_snapshot"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("title", sa.Text(), nullable=True))
    op.execute(
        sa.text("UPDATE tasks SET title = instruction WHERE title IS NULL")
    )


def downgrade() -> None:
    op.drop_column("tasks", "title")
