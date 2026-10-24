import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)


class UserView(BaseModel):
    id: uuid.UUID
    email: str
    role: Literal["admin", "user"]
    must_change_password: bool

    model_config = ConfigDict(from_attributes=True)


class LoginResponse(BaseModel):
    user: UserView
    csrf_token: str


class CreateUserRequest(BaseModel):
    email: EmailStr
    initial_password: str = Field(min_length=12, max_length=1024)


class PasswordResetRequest(BaseModel):
    initial_password: str = Field(min_length=12, max_length=1024)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=12, max_length=1024)


class AccessKeyResponse(BaseModel):
    access_key: str
    created_at: datetime


class AdminTaskSummary(BaseModel):
    id: uuid.UUID
    owner_email: str
    status: str
    created_at: datetime
    updated_at: datetime
    error: dict | None


class AdminSettingsUpdate(BaseModel):
    manager_model: str = Field(min_length=1, max_length=200)
    worker_model: str = Field(min_length=1, max_length=200)
    max_concurrent_agents: int = Field(ge=1, le=32)
    max_agent_attempts: int = Field(ge=1, le=3)
    max_quality_revisions_per_subtask: int = Field(ge=0, le=5)
    max_quality_revisions_per_task: int = Field(ge=0, le=25)


class AdminSettingsView(AdminSettingsUpdate):
    openai_key_configured: bool
    tavily_key_configured: bool
    version: int


class OpenAIKeyRequest(BaseModel):
    api_key: SecretStr = Field(min_length=1, max_length=1024)


class TavilyKeyRequest(BaseModel):
    api_key: SecretStr = Field(min_length=1, max_length=1024)


class TaskCreate(BaseModel):
    instruction: str = Field(min_length=1, max_length=20_000)


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)


class ToolBudgetApprovalDecision(BaseModel):
    request_id: uuid.UUID
    approved: bool


class MessageView(BaseModel):
    id: uuid.UUID
    sequence: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TaskView(BaseModel):
    id: uuid.UUID
    instruction: str
    title: str | None
    status: str
    created_at: datetime
    updated_at: datetime
    result: str | None
    error: dict | None
    completion_report: dict | None = None
    pending_tool_approval: dict | None = None

    model_config = ConfigDict(from_attributes=True)


class TaskFileView(BaseModel):
    id: uuid.UUID
    kind: Literal["file"] = "file"
    path: str = Field(validation_alias="workspace_rel_path")
    role: Literal["input", "source", "intermediate", "final"]
    source_kind: str
    source_ref: str | None
    size_bytes: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class WorkspaceDirectoryView(BaseModel):
    kind: Literal["directory"] = "directory"
    path: str


class TaskDeletionView(BaseModel):
    task_id: uuid.UUID
    status: Literal["pending", "leased"]
