import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SystemSettings(Base):
    __tablename__ = "system_settings"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_system_settings_singleton"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    settings_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    openai_key_configured: Mapped[bool] = mapped_column(Boolean, default=False)
    tavily_key_configured: Mapped[bool] = mapped_column(Boolean, default=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ApiAccessKey(Base):
    __tablename__ = "api_access_keys"
    __table_args__ = (
        Index(
            "uq_api_access_keys_active_user",
            "user_id",
            unique=True,
            postgresql_where=text("revoked_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE")
    )
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class WebSession(Base):
    __tablename__ = "web_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    session_token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    csrf_token_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    absolute_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True)
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_owner_updated", "owner_id", "updated_at"),
        CheckConstraint(
            "status IN ('queued', 'planning', 'running', 'waiting_approval', "
            "'succeeded', 'partially_completed', 'failed', 'cancelled', "
            "'needs_attention')",
            name="ck_tasks_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    instruction: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(32), default="queued", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    result: Mapped[str | None] = mapped_column(Text)
    error: Mapped[dict | None] = mapped_column(JSONB)
    completion_report: Mapped[dict | None] = mapped_column(JSONB)
    pending_tool_approval: Mapped[dict | None] = mapped_column(JSONB)
    config_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict)
    deletion_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    messages: Mapped[list["ConversationMessage"]] = relationship(
        cascade="all, delete-orphan"
    )


class TaskFile(Base):
    __tablename__ = "task_files"
    __table_args__ = (
        UniqueConstraint(
            "task_id", "workspace_rel_path", name="uq_task_files_path"
        ),
        CheckConstraint(
            "role IN ('input', 'source', 'intermediate', 'final')",
            name="ck_task_files_role",
        ),
        Index("ix_task_files_task_path", "task_id", "workspace_rel_path"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), index=True
    )
    workspace_rel_path: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(16), default="input")
    source_kind: Mapped[str] = mapped_column(String(32), default="user_upload")
    source_ref: Mapped[str | None] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        Index("uq_messages_task_sequence", "task_id", "sequence", unique=True),
        CheckConstraint(
            "role IN ('user', 'assistant')", name="ck_message_role"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Subtask(Base):
    __tablename__ = "subtasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'ready', 'running', 'waiting_approval', "
            "'reviewing', 'revising', 'accepted', 'partially_accepted', "
            "'succeeded', 'failed', 'blocked', 'cancelled')",
            name="ck_subtasks_status",
        ),
        CheckConstraint(
            "tool_calls_used >= 0", name="ck_subtasks_tool_calls_used"
        ),
        CheckConstraint(
            "tool_calls_limit IS NULL OR "
            "tool_calls_limit >= 0",
            name="ck_subtasks_tool_calls_limit",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(300))
    instruction: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    dependencies: Mapped[list] = mapped_column(JSONB, default=list)
    model_id: Mapped[str] = mapped_column(String(200))
    skill_names: Mapped[list] = mapped_column(JSONB, default=list)
    plan_position: Mapped[int] = mapped_column(Integer, default=0)
    artifact_goal: Mapped[str] = mapped_column(Text, default="")
    acceptance_checklist: Mapped[list] = mapped_column(JSONB, default=list)
    plan_version: Mapped[int | None] = mapped_column(Integer)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    quality_revision_count: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls_used: Mapped[int] = mapped_column(
        BigInteger, default=0, server_default="0"
    )
    tool_calls_limit: Mapped[int | None] = mapped_column(BigInteger)
    result: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TaskPlanRecord(Base):
    __tablename__ = "task_plans"
    __table_args__ = (
        UniqueConstraint("task_id", "version", name="uq_task_plans_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    conversation_sequence: Mapped[int] = mapped_column(Integer)
    summary: Mapped[str] = mapped_column(Text)
    final_checklist: Mapped[list] = mapped_column(JSONB)
    change_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class QualityReview(Base):
    __tablename__ = "quality_reviews"
    __table_args__ = (
        CheckConstraint(
            "target_type IN ('subtask', 'final')",
            name="ck_quality_reviews_target_type",
        ),
        CheckConstraint(
            "verdict IN ('accepted', 'revision_requested', "
            "'partially_accepted')",
            name="ck_quality_reviews_verdict",
        ),
        Index("ix_quality_reviews_task_plan", "task_id", "plan_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), index=True
    )
    plan_version: Mapped[int] = mapped_column(Integer)
    target_type: Mapped[str] = mapped_column(String(16))
    target_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    revision_round: Mapped[int] = mapped_column(Integer, default=0)
    checklist_results: Mapped[list] = mapped_column(JSONB)
    verdict: Mapped[str] = mapped_column(String(32))
    reviewer_model_id: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index(
            "ix_jobs_available", "status", "available_at", "lease_expires_at"
        ),
        CheckConstraint(
            "status IN ('pending', 'leased', 'completed', 'failed')",
            name="ck_jobs_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(16), default="pending")
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TaskDeletionJob(Base):
    __tablename__ = "task_deletion_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'leased')",
            name="ck_task_deletion_jobs_status",
        ),
        Index(
            "ix_task_deletion_jobs_available",
            "status",
            "available_at",
            "lease_expires_at",
        ),
    )

    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(16), default="pending")
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
