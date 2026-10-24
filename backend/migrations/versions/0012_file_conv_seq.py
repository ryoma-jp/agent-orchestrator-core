"""Associate uploaded inputs with their conversation position."""

from alembic import op
import sqlalchemy as sa

revision = "0012_file_conv_seq"
down_revision = "0011_task_title_generated"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "task_files",
        sa.Column("conversation_sequence", sa.Integer(), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE task_files
            SET conversation_sequence = (
                SELECT MIN(conversation_messages.sequence)
                FROM conversation_messages
                WHERE conversation_messages.task_id = task_files.task_id
                  AND conversation_messages.role = 'user'
                  AND conversation_messages.content =
                      'Attached input file: ' || task_files.workspace_rel_path
            )
            WHERE task_files.role = 'input'
              AND task_files.source_kind = 'user_upload'
            """
        )
    )


def downgrade() -> None:
    op.drop_column("task_files", "conversation_sequence")
