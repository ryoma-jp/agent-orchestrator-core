"""Track whether the Tavily API key has been configured."""

from alembic import op
import sqlalchemy as sa

revision = "0003_tavily_key_configured"
down_revision = "0002_quality_reviews"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "system_settings",
        sa.Column(
            "tavily_key_configured",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("system_settings", "tavily_key_configured")
