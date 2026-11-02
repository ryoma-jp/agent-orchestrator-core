"""Persist specialist Agent selections and review provenance."""

from alembic import op
import sqlalchemy as sa

revision = "0016_specialist_agents"
down_revision = "0015_image_generation_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "subtasks",
        sa.Column("agent_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "subtasks",
        sa.Column(
            "agent_definition_digest", sa.String(length=64), nullable=True
        ),
    )
    op.add_column(
        "quality_reviews",
        sa.Column(
            "reviewer_model_provider", sa.String(length=16), nullable=True
        ),
    )
    op.add_column(
        "quality_reviews",
        sa.Column("reviewer_agent_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "quality_reviews",
        sa.Column(
            "reviewer_agent_digest", sa.String(length=64), nullable=True
        ),
    )
    op.add_column(
        "quality_reviews",
        sa.Column(
            "specialist_model_provider", sa.String(length=16), nullable=True
        ),
    )
    op.add_column(
        "quality_reviews",
        sa.Column("specialist_model_id", sa.String(length=200), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("quality_reviews", "reviewer_agent_digest")
    op.drop_column("quality_reviews", "reviewer_agent_id")
    op.drop_column("quality_reviews", "reviewer_model_provider")
    op.drop_column("subtasks", "agent_definition_digest")
    op.drop_column("subtasks", "agent_id")
    op.drop_column("quality_reviews", "specialist_model_id")
    op.drop_column("quality_reviews", "specialist_model_provider")
