"""Record image generation seed and quality settings."""

from alembic import op
import sqlalchemy as sa

revision = "0015_image_generation_settings"
down_revision = "0014_openai_image_generation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_generation_revisions",
        sa.Column("seed", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "image_generation_revisions",
        sa.Column("quality", sa.String(length=40), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("image_generation_revisions", "quality")
    op.drop_column("image_generation_revisions", "seed")
