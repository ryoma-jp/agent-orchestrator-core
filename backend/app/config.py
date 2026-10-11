from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = (
        "postgresql+asyncpg://orchestrator:orchestrator@"
        "localhost:5432/orchestrator"
    )
    manager_model: str
    worker_model: str
    initial_admin_email: str | None = None
    initial_admin_password: SecretStr | None = None
    session_cookie_secure: bool = False
    session_idle_minutes: int = 30
    session_absolute_hours: int = 12
    session_poll_seconds: float = 1.0
    max_concurrent_agents: int = 4
    max_agent_attempts: int = 3
    job_lease_seconds: int = 60
    agent_timeout_seconds: int = 180
    openai_base_url: str | None = None
    vault_addr: str | None = None
    vault_kv_mount: str = "secret"
    vault_openai_key_path: str = "agent-orchestrator/openai"
    vault_api_role_id_file: str | None = None
    vault_api_secret_id_file: str | None = None
    vault_worker_role_id_file: str | None = None
    vault_worker_secret_id_file: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
