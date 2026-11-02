"""Shared prompt and OpenAI image generation support."""

import asyncio
import base64
import binascii
import re
import uuid
from collections.abc import Awaitable, Callable
from urllib.parse import quote

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent_schemas import ImageGenerationPrompt
from app.config import get_settings
from app.tools import ToolRegistration
from app.vault import read_openai_api_key

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SAFE_FILE_COMPONENT = re.compile(r"[^/\\\x00-\x1f\x7f]+\Z")
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


class ImageGenerationArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(
        default="primary",
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )
    image_prompt: ImageGenerationPrompt
    filename: str = Field(
        default="generated-image.png", min_length=5, max_length=220
    )
    width: int = Field(default=1024, ge=256, le=3840)
    height: int = Field(default=1024, ge=256, le=3840)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if (
            _SAFE_FILE_COMPONENT.fullmatch(value) is None
            or value in {".", ".."}
            or not value.casefold().endswith(".png")
        ):
            raise ValueError("image_filename_invalid")
        return value


class OpenAIImageGenerationArguments(ImageGenerationArguments):
    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, value: int) -> int:
        if value % 16:
            raise ValueError("image_dimension_must_be_multiple_of_16")
        return value

    def validate_image_size(self) -> None:
        width = self.width
        height = self.height
        if (
            max(width, height) / min(width, height) > 3
            or not 655_360 <= width * height <= 8_294_400
        ):
            raise ValueError("openai_image_size_unsupported")


def build_openai_image_prompt(image_prompt: ImageGenerationPrompt) -> str:
    positive_prompt = image_prompt.positive_prompt
    if not image_prompt.negative_prompt:
        return positive_prompt
    return (
        f"{positive_prompt}\n\nDo not include: "
        f"{image_prompt.negative_prompt}"
    )


async def save_generated_png(
    task_id: uuid.UUID,
    filename: str,
    image_bytes: bytes,
    before_workspace_write: Callable[[], Awaitable[None]],
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    settings = get_settings()
    if settings.workspace_internal_token is None:
        raise RuntimeError("workspace_artifact_persistence_unavailable")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise RuntimeError("image_too_large")
    if not image_bytes.startswith(_PNG_SIGNATURE):
        raise RuntimeError("invalid_png")

    suffix = uuid.uuid4().hex[:8]
    relative_path = f"artifacts/{filename[:-4]}-{suffix}.png"
    await before_workspace_write()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60), transport=transport
    ) as client:
        response = await client.put(
            f"{settings.api_internal_base_url.rstrip('/')}/api/tasks/"
            f"{task_id}/workspace/internal/"
            f"{quote(relative_path, safe='/')}",
            params={"role": "final"},
            headers={
                "X-Workspace-Token": (
                    settings.workspace_internal_token.get_secret_value()
                )
            },
            content=image_bytes,
        )
        response.raise_for_status()
    return relative_path


class OpenAIImageGenerationTool:
    def __init__(
        self,
        task_id: uuid.UUID,
        model: str,
        before_workspace_write: Callable[[], Awaitable[None]],
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.task_id = task_id
        self.model = model
        self.before_workspace_write = before_workspace_write
        self.transport = transport

    def registration(self) -> ToolRegistration:
        settings = get_settings()
        return ToolRegistration(
            name="openai_generate_image",
            version="1",
            description=(
                "Generate a user-requested PNG image with the configured "
                "OpenAI image model and save it under this task's artifacts "
                "directory. Pass the exact two-key "
                "positive_prompt/negative_prompt JSON as image_prompt. "
                "Use only for explicit image requests."
            ),
            risk_class="writes_data",
            timeout_seconds=max(settings.agent_timeout_seconds, 180) + 30,
            arguments_model=OpenAIImageGenerationArguments,
            handler=self._generate,
            openai_parameters={
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "image_prompt": {
                        "type": "object",
                        "properties": {
                            "positive_prompt": {"type": "string"},
                            "negative_prompt": {"type": "string"},
                        },
                        "required": [
                            "positive_prompt",
                            "negative_prompt",
                        ],
                        "additionalProperties": False,
                    },
                    "filename": {"type": "string"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                },
                "required": [
                    "image_id",
                    "image_prompt",
                    "filename",
                    "width",
                    "height",
                ],
                "additionalProperties": False,
            },
        )

    async def _generate(
        self, raw_arguments: BaseModel
    ) -> dict[str, object]:
        if not isinstance(raw_arguments, OpenAIImageGenerationArguments):
            raise TypeError("invalid_openai_image_arguments")
        try:
            raw_arguments.validate_image_size()
        except ValueError:
            return {"error": "openai_image_size_unsupported"}

        try:
            api_key = await asyncio.to_thread(read_openai_api_key)
        except Exception:
            return {"error": "openai_image_api_key_unavailable"}

        settings = get_settings()
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=settings.openai_base_url or "https://api.openai.com/v1",
            max_retries=0,
            timeout=settings.agent_timeout_seconds,
        )
        try:
            response = await client.images.generate(
                model=self.model,
                prompt=build_openai_image_prompt(raw_arguments.image_prompt),
                size=f"{raw_arguments.width}x{raw_arguments.height}",
                quality="auto",
                output_format="png",
                timeout=settings.agent_timeout_seconds,
            )
        except APITimeoutError:
            return {"error": "openai_image_generation_timeout"}
        except APIConnectionError:
            return {"error": "openai_image_provider_unavailable"}
        except APIStatusError as exc:
            return {"error": self._api_error_code(exc.status_code)}
        finally:
            await client.close()

        image_data = response.data[0].b64_json if response.data else None
        if not isinstance(image_data, str):
            return {"error": "openai_image_response_missing_data"}
        max_encoded_bytes = ((_MAX_IMAGE_BYTES + 2) // 3) * 4
        if len(image_data) > max_encoded_bytes:
            return {"error": "openai_image_too_large"}
        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except (ValueError, binascii.Error):
            return {"error": "openai_image_response_invalid"}
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            return {"error": "openai_image_too_large"}
        if not image_bytes.startswith(_PNG_SIGNATURE):
            return {"error": "openai_image_response_invalid"}

        try:
            artifact_path = await save_generated_png(
                self.task_id,
                raw_arguments.filename,
                image_bytes,
                self.before_workspace_write,
                self.transport,
            )
        except Exception:
            return {"error": "openai_image_workspace_write_failed"}
        return {
            "artifact_path": artifact_path,
            "image_id": raw_arguments.image_id,
            "width": raw_arguments.width,
            "height": raw_arguments.height,
            "seed": None,
            "quality": "auto",
            "format": "png",
        }

    @staticmethod
    def _api_error_code(status_code: int) -> str:
        if status_code in {401, 403}:
            return "openai_image_authentication_failed"
        if status_code == 404:
            return "openai_image_model_unavailable"
        if status_code == 429:
            return "openai_image_rate_limited"
        if status_code >= 500:
            return "openai_image_provider_unavailable"
        return "openai_image_request_failed"
