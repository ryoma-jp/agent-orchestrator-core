"""Add OpenAI image generation history fields."""

from alembic import op
import sqlalchemy as sa

revision = "0014_openai_image_generation"
down_revision = "0013_image_generation_revisions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "image_generation_revisions",
        "steps",
        existing_type=sa.Integer(),
        nullable=True,
    )
    op.add_column(
        "image_generation_revisions",
        sa.Column("openai_prompt", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM image_generation_revisions
                    WHERE steps IS NULL
                ) THEN
                    RAISE EXCEPTION
                        'Cannot require steps while OpenAI revisions exist';
                END IF;
            END;
            $$
            """
        )
    )
    op.drop_column("image_generation_revisions", "openai_prompt")
    op.alter_column(
        "image_generation_revisions",
        "steps",
        existing_type=sa.Integer(),
        nullable=False,
    )
