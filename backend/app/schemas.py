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


class AdminSettingsView(AdminSettingsUpdate):
    openai_key_configured: bool
    version: int


class OpenAIKeyRequest(BaseModel):
    api_key: SecretStr = Field(min_length=1, max_length=1024)


class TaskCreate(BaseModel):
    instruction: str = Field(min_length=1, max_length=20_000)


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)


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
    status: str
    created_at: datetime
    updated_at: datetime
    result: str | None
    error: dict | None

    model_config = ConfigDict(from_attributes=True)
