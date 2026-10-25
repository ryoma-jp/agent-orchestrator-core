import asyncio

import httpx

from app.config import OLLAMA_BASE_URL, get_settings
from app.schemas import ModelProvider
from app.vault import read_openai_model_catalog_key


class ModelCatalogError(RuntimeError):
    pass


async def list_models(provider: ModelProvider) -> list[str]:
    settings = get_settings()
    if provider == "openai":
        try:
            api_key = await asyncio.to_thread(read_openai_model_catalog_key)
        except Exception:
            raise ModelCatalogError(
                "openai_model_catalog_unavailable"
            ) from None
        base_url = settings.openai_base_url or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
    else:
        url = f"{OLLAMA_BASE_URL}/api/tags"
        headers = {}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        raise ModelCatalogError(
            f"{provider}_model_catalog_unavailable"
        ) from None

    if provider == "openai":
        entries = payload.get("data") if isinstance(payload, dict) else None
        entries = entries if isinstance(entries, list) else []
        identifiers = [
            item.get("id") for item in entries if isinstance(item, dict)
        ]
    else:
        entries = payload.get("models") if isinstance(payload, dict) else None
        entries = entries if isinstance(entries, list) else []
        identifiers = [
            item.get("name") for item in entries if isinstance(item, dict)
        ]
    return sorted(
        {
            identifier
            for identifier in identifiers
            if isinstance(identifier, str)
        }
    )
