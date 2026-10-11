import hvac

from app.config import get_settings


def _authenticated_client(
    role_id_file: str | None, secret_id_file: str | None
):
    settings = get_settings()
    if not settings.vault_addr:
        raise RuntimeError("vault_not_configured")
    if not role_id_file or not secret_id_file:
        raise RuntimeError("vault_approle_credentials_not_configured")

    with open(role_id_file, encoding="utf-8") as role_file:
        role_id = role_file.read().strip()
    with open(secret_id_file, encoding="utf-8") as secret_file:
        secret_id = secret_file.read().strip()
    if not role_id or not secret_id:
        raise RuntimeError("vault_approle_credentials_empty")

    client = hvac.Client(url=settings.vault_addr)
    client.auth.approle.login(role_id=role_id, secret_id=secret_id)
    if not client.is_authenticated():
        raise RuntimeError("vault_authentication_failed")
    return client


def read_openai_api_key() -> str:
    settings = get_settings()
    try:
        client = _authenticated_client(
            settings.vault_worker_role_id_file,
            settings.vault_worker_secret_id_file,
        )
        result = client.secrets.kv.v2.read_secret_version(
            mount_point=settings.vault_kv_mount,
            path=settings.vault_openai_key_path,
        )
        api_key = result["data"]["data"].get("api_key")
        if not isinstance(api_key, str) or not api_key:
            raise RuntimeError("openai_api_key_not_configured")
        return api_key
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("vault_key_read_failed") from exc


def write_openai_api_key(api_key: str) -> None:
    settings = get_settings()
    if not api_key:
        raise RuntimeError("openai_api_key_empty")
    try:
        client = _authenticated_client(
            settings.vault_api_role_id_file,
            settings.vault_api_secret_id_file,
        )
        client.secrets.kv.v2.create_or_update_secret(
            mount_point=settings.vault_kv_mount,
            path=settings.vault_openai_key_path,
            secret={"api_key": api_key},
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("vault_key_write_failed") from exc
