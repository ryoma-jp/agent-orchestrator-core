"""Store append-only image generation prompt history."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0013_image_generation_revisions"
down_revision = "0012_file_conv_seq"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "image_generation_revisions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column(
            "task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "subtask_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("subtasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("plan_version", sa.Integer(), nullable=True),
        sa.Column("conversation_sequence", sa.Integer(), nullable=True),
        sa.Column("generation_no", sa.Integer(), nullable=False),
        sa.Column(
            "trigger_quality_review_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("quality_reviews.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "source_artifact_paths",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column(
            "negative_prompt",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "model_id",
            sa.String(length=200),
            nullable=False,
        ),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("steps", sa.Integer(), nullable=False),
        sa.Column(
            "unmet_items",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "is_backfilled",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "generation_no >= 1",
            name="ck_image_generation_revisions_generation_no",
        ),
        sa.UniqueConstraint(
            "subtask_id",
            "generation_no",
            name="uq_image_generation_revisions_subtask_generation",
        ),
    )
    op.create_index(
        "ix_image_generation_revisions_task_id",
        "image_generation_revisions",
        ["task_id"],
    )
    op.create_index(
        "ix_image_generation_revisions_subtask_id",
        "image_generation_revisions",
        ["subtask_id"],
    )
    op.create_index(
        "ix_image_generation_revisions_review_id",
        "image_generation_revisions",
        ["trigger_quality_review_id"],
    )
    op.create_index(
        "ix_image_generation_revisions_task_created",
        "image_generation_revisions",
        ["task_id", "created_at", "id"],
    )
    op.execute(
        sa.text(
            """
            INSERT INTO image_generation_revisions (
                id,
                task_id,
                subtask_id,
                plan_version,
                generation_no,
                source_artifact_paths,
                artifact_path,
                prompt,
                negative_prompt,
                model_id,
                width,
                height,
                steps,
                unmet_items,
                is_backfilled
            )
            SELECT
                gen_random_uuid(),
                subtask.task_id,
                subtask.id,
                subtask.plan_version,
                generated.ordinality::integer,
                '[]'::jsonb,
                generated.value->>'artifact_path',
                generated.value->>'prompt',
                COALESCE(generated.value->>'negative_prompt', ''),
                COALESCE(task.config_snapshot->>'worker_image_model', ''),
                (generated.value->>'width')::integer,
                (generated.value->>'height')::integer,
                (generated.value->>'steps')::integer,
                '[]'::jsonb,
                true
            FROM subtasks AS subtask
            JOIN tasks AS task ON task.id = subtask.task_id
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(
                        subtask.result->'image_generations'
                    ) = 'array'
                    THEN subtask.result->'image_generations'
                    ELSE '[]'::jsonb
                END
            ) WITH ORDINALITY AS generated(value, ordinality)
            WHERE generated.value->>'artifact_path'
                    ~ '^artifacts/[^/]+[.]png$'
              AND jsonb_typeof(generated.value->'prompt') = 'string'
              AND COALESCE(generated.value->>'width', '') ~ '^[0-9]+$'
              AND COALESCE(generated.value->>'height', '') ~ '^[0-9]+$'
              AND COALESCE(generated.value->>'steps', '') ~ '^[0-9]+$'
              AND (generated.value->>'width')::integer
                    BETWEEN 256 AND 1536
              AND (generated.value->>'height')::integer
                    BETWEEN 256 AND 1536
              AND (generated.value->>'steps')::integer BETWEEN 1 AND 60
            """
        )
    )


def downgrade() -> None:
    op.drop_index(
        "ix_image_generation_revisions_task_created",
        table_name="image_generation_revisions",
    )
    op.drop_index(
        "ix_image_generation_revisions_review_id",
        table_name="image_generation_revisions",
    )
    op.drop_index(
        "ix_image_generation_revisions_subtask_id",
        table_name="image_generation_revisions",
    )
    op.drop_index(
        "ix_image_generation_revisions_task_id",
        table_name="image_generation_revisions",
    )
    op.drop_table("image_generation_revisions")
