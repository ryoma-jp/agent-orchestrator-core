import asyncio

import httpx

from app.config import OLLAMA_BASE_URL, get_settings
from app.schemas import ModelCatalogProvider
from app.vault import read_openai_model_catalog_key


class ModelCatalogError(RuntimeError):
    pass


async def list_models(provider: ModelCatalogProvider) -> list[str]:
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
    elif provider == "ollama":
        url = f"{OLLAMA_BASE_URL}/api/tags"
        headers = {}
    else:
        url = (
            f"{settings.comfyui_base_url.rstrip('/')}/object_info/"
            "CheckpointLoaderSimple"
        )
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
    elif provider == "ollama":
        entries = payload.get("models") if isinstance(payload, dict) else None
        entries = entries if isinstance(entries, list) else []
        identifiers = [
            item.get("name") for item in entries if isinstance(item, dict)
        ]
    else:
        node = (
            payload.get("CheckpointLoaderSimple")
            if isinstance(payload, dict)
            else None
        )
        node_input = node.get("input") if isinstance(node, dict) else None
        required = (
            node_input.get("required")
            if isinstance(node_input, dict)
            else None
        )
        checkpoint_options = (
            required.get("ckpt_name")
            if isinstance(required, dict)
            else None
        )
        identifiers = (
            checkpoint_options[0]
            if isinstance(checkpoint_options, list)
            and checkpoint_options
            and isinstance(checkpoint_options[0], list)
            else []
        )
        if not isinstance(identifiers, list):
            raise ModelCatalogError("comfyui_model_catalog_invalid")
    return sorted(
        {
            identifier
            for identifier in identifiers
            if isinstance(identifier, str)
        }
    )
