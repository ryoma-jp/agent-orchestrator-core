"""Bounded ComfyUI image generation for task-scoped artifacts."""

import asyncio
import re
import uuid
from collections.abc import Awaitable, Callable
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings
from app.tools import ToolRegistration

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SAFE_FILE_COMPONENT = re.compile(r"[^/\\\x00-\x1f\x7f]+\Z")


class ComfyUIImageArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4_000)
    negative_prompt: str = Field(default="", max_length=2_000)
    filename: str = Field(
        default="generated-image.png", min_length=5, max_length=220
    )
    width: int = Field(default=1024, ge=256, le=1536)
    height: int = Field(default=1024, ge=256, le=1536)
    steps: int = Field(default=25, ge=1, le=60)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("image_prompt_empty")
        return value

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

    @field_validator("width", "height")
    @classmethod
    def validate_dimension(cls, value: int) -> int:
        if value % 8:
            raise ValueError("image_dimension_must_be_multiple_of_8")
        return value


class ComfyUIImageGenerationTool:
    def __init__(
        self,
        task_id: uuid.UUID,
        checkpoint_name: str,
        before_workspace_write: Callable[[], Awaitable[None]],
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.task_id = task_id
        self.checkpoint_name = checkpoint_name
        self.before_workspace_write = before_workspace_write
        self.transport = transport

    def registration(self) -> ToolRegistration:
        settings = get_settings()
        return ToolRegistration(
            name="comfyui_generate_image",
            version="1",
            description=(
                "Generate a user-requested PNG image with ComfyUI and save "
                "it under this task's artifacts directory. Use only for "
                "explicit image requests."
            ),
            risk_class="writes_data",
            timeout_seconds=settings.comfyui_timeout_seconds + 60,
            arguments_model=ComfyUIImageArguments,
            handler=self._generate,
            openai_parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "filename": {"type": "string"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "steps": {"type": "integer"},
                },
                "required": [
                    "prompt",
                    "negative_prompt",
                    "filename",
                    "width",
                    "height",
                    "steps",
                ],
                "additionalProperties": False,
            },
        )

    async def _generate(
        self, raw_arguments: BaseModel
    ) -> dict[str, object]:
        if not isinstance(raw_arguments, ComfyUIImageArguments):
            raise TypeError("invalid_comfyui_arguments")
        settings = get_settings()
        checkpoint_name = self._validate_checkpoint_name(
            self.checkpoint_name
        )
        workspace_token = settings.workspace_internal_token
        if workspace_token is None:
            raise RuntimeError("workspace_artifact_persistence_unavailable")

        suffix = uuid.uuid4().hex[:8]
        filename = f"{raw_arguments.filename[:-4]}-{suffix}.png"
        filename_prefix = f"task-{self.task_id.hex}-{suffix}"
        workflow = self._workflow(
            raw_arguments, checkpoint_name, filename_prefix
        )
        base_url = settings.comfyui_base_url.rstrip("/")
        timeout_seconds = settings.comfyui_timeout_seconds

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10), transport=self.transport
        ) as client:
            prompt_response = await client.post(
                f"{base_url}/prompt",
                json={"prompt": workflow},
                timeout=min(30, timeout_seconds),
            )
            prompt_response.raise_for_status()
            prompt_payload = prompt_response.json()
            prompt_id = (
                prompt_payload.get("prompt_id")
                if isinstance(prompt_payload, dict)
                else None
            )
            if not isinstance(prompt_id, str) or not prompt_id:
                raise RuntimeError("comfyui_invalid_prompt_response")

            image = await self._wait_for_image(
                client,
                base_url,
                prompt_id,
                filename_prefix,
                timeout_seconds,
            )
            image_bytes = await self._download_png(
                client, base_url, image, settings.comfyui_max_image_bytes
            )

            await self.before_workspace_write()
            relative_path = f"artifacts/{filename}"
            response = await client.put(
                f"{settings.api_internal_base_url.rstrip('/')}/api/tasks/"
                f"{self.task_id}/workspace/internal/"
                f"{quote(relative_path, safe='/')}",
                params={"role": "final"},
                headers={
                    "X-Workspace-Token": workspace_token.get_secret_value()
                },
                content=image_bytes,
            )
            response.raise_for_status()

        return {
            "artifact_path": relative_path,
            "width": raw_arguments.width,
            "height": raw_arguments.height,
            "format": "png",
        }

    async def _wait_for_image(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        prompt_id: str,
        filename_prefix: str,
        timeout_seconds: int,
    ) -> dict[str, str]:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise RuntimeError("comfyui_generation_timeout")
            response = await client.get(
                f"{base_url}/history/{prompt_id}",
                timeout=min(10, remaining),
            )
            response.raise_for_status()
            payload = response.json()
            prompt = (
                payload.get(prompt_id)
                if isinstance(payload, dict)
                else None
            )
            if isinstance(prompt, dict):
                status = prompt.get("status")
                if (
                    isinstance(status, dict)
                    and status.get("status_str") == "error"
                ):
                    raise RuntimeError("comfyui_generation_failed")
                outputs = prompt.get("outputs")
                if isinstance(outputs, dict):
                    for output in outputs.values():
                        images = (
                            output.get("images")
                            if isinstance(output, dict)
                            else None
                        )
                        if not isinstance(images, list):
                            continue
                        for image in images:
                            if not isinstance(image, dict):
                                continue
                            filename = image.get("filename")
                            subfolder = image.get("subfolder", "")
                            image_type = image.get("type")
                            if (
                                isinstance(filename, str)
                                and filename.startswith(filename_prefix)
                                and _SAFE_FILE_COMPONENT.fullmatch(filename)
                                and subfolder == ""
                                and image_type == "output"
                            ):
                                return {"filename": filename}
            await asyncio.sleep(min(1, max(remaining, 0)))

    async def _download_png(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        image: dict[str, str],
        max_image_bytes: int,
    ) -> bytes:
        content = bytearray()
        async with client.stream(
            "GET",
            f"{base_url}/view",
            params={
                "filename": image["filename"],
                "subfolder": "",
                "type": "output",
            },
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(
                ";", 1
            )[0]
            if content_type not in {
                "image/png",
                "application/octet-stream",
            }:
                raise RuntimeError("comfyui_invalid_image_type")
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > max_image_bytes:
                    raise RuntimeError("comfyui_image_too_large")
        if not content.startswith(_PNG_SIGNATURE):
            raise RuntimeError("comfyui_invalid_png")
        return bytes(content)

    @staticmethod
    def _validate_checkpoint_name(value: str) -> str:
        if (
            not value
            or _SAFE_FILE_COMPONENT.fullmatch(value) is None
            or value in {".", ".."}
        ):
            raise RuntimeError("comfyui_checkpoint_not_configured")
        return value

    @staticmethod
    def _workflow(
        arguments: ComfyUIImageArguments,
        checkpoint_name: str,
        filename_prefix: str,
    ) -> dict[str, dict[str, object]]:
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint_name},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": arguments.prompt, "clip": ["1", 1]},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": arguments.negative_prompt,
                    "clip": ["1", 1],
                },
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": arguments.width,
                    "height": arguments.height,
                    "batch_size": 1,
                },
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": uuid.uuid4().int % (2**63),
                    "steps": arguments.steps,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": filename_prefix,
                    "images": ["6", 0],
                },
            },
        }
