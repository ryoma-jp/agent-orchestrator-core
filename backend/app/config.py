from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

OLLAMA_BASE_URL = "http://ollama:11434"


class Settings(BaseSettings):
    database_url: str = (
        "postgresql+asyncpg://orchestrator:orchestrator@"
        "localhost:5432/orchestrator"
    )
    manager_model_provider: Literal["openai", "ollama"] = "openai"
    manager_model: str = "gpt-6.1-sol"
    worker_model_provider: Literal["openai", "ollama"] = "openai"
    worker_model: str = "gpt-6-luna"
    worker_image_provider: Literal["openai", "ollama", "comfyui"] = (
        "openai"
    )
    worker_image_model: str = "gpt-image-2.5-sunburst"
    initial_admin_email: str | None = None
    initial_admin_password: SecretStr | None = None
    session_cookie_secure: bool = False
    session_idle_minutes: int = 30
    session_absolute_hours: int = 12
    session_poll_seconds: float = 1.0
    max_concurrent_agents: int = 4
    max_agent_attempts: int = 3
    max_agent_turns: int = Field(default=24, ge=1, le=64)
    max_tool_calls_per_subtask: int = Field(default=20, ge=0, le=100)
    max_quality_revisions_per_subtask: int = 2
    max_quality_revisions_per_task: int = 10
    job_lease_seconds: int = 60
    agent_timeout_seconds: int = 180
    workspace_artifact_timeout_seconds: int = Field(
        default=60, ge=1, le=300
    )
    workspace_artifact_max_attempts: int = Field(default=3, ge=1, le=5)
    tavily_timeout_seconds: int = 20
    tavily_search_skill_path: str = (
        "/opt/agent-skills/tavily-search/SKILL.md"
    )
    comfyui_enabled: bool = False
    comfyui_base_url: str = "http://comfyui:8188"
    comfyui_timeout_seconds: int = Field(default=300, ge=10, le=1800)
    comfyui_max_image_bytes: int = Field(
        default=20 * 1024 * 1024, ge=1_048_576, le=100 * 1024 * 1024
    )
    private_skills_path: str = "/opt/private-agent-skills"
    openai_base_url: str | None = None
    vault_addr: str | None = None
    vault_kv_mount: str = "secret"
    vault_openai_key_path: str = "agent-orchestrator/openai"
    vault_tavily_key_path: str = "agent-orchestrator/tavily"
    vault_api_role_id_file: str | None = None
    vault_api_secret_id_file: str | None = None
    vault_model_catalog_role_id_file: str | None = None
    vault_model_catalog_secret_id_file: str | None = None
    vault_worker_role_id_file: str | None = None
    vault_worker_secret_id_file: str | None = None
    workspace_root: str = ".data/workspaces"
    workspace_internal_token: SecretStr | None = None
    api_internal_base_url: str = "http://api:8000"

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
