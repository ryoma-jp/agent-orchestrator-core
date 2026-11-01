import base64
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from app import image_generation as image_generation_module
from app.agent_schemas import ImageGenerationPrompt
from app.image_generation import (
    OpenAIImageGenerationArguments,
    OpenAIImageGenerationTool,
    build_openai_image_prompt,
    save_generated_png,
)

_PNG = b"\x89PNG\r\n\x1a\nimage-bytes"


def test_openai_image_prompt_combines_negative_prompt_only_when_present():
    assert build_openai_image_prompt(
        ImageGenerationPrompt(
            positive_prompt="A paper crane",
            negative_prompt="blurry",
        )
    ) == "A paper crane\n\nDo not include: blurry"
    assert build_openai_image_prompt(
        ImageGenerationPrompt(
            positive_prompt="A paper crane",
            negative_prompt="",
        )
    ) == "A paper crane"


def test_openai_image_arguments_reject_invalid_sizes_and_paths():
    with pytest.raises(ValidationError):
        OpenAIImageGenerationArguments(
            image_prompt={"positive_prompt": "image", "negative_prompt": ""},
            filename="../image.png",
        )
    with pytest.raises(ValidationError):
        OpenAIImageGenerationArguments(
            image_prompt={"positive_prompt": "image", "negative_prompt": ""},
            width=1025,
        )
    too_small = OpenAIImageGenerationArguments(
        image_prompt={"positive_prompt": "image", "negative_prompt": ""},
        width=512,
        height=512,
    )
    with pytest.raises(ValueError, match="openai_image_size_unsupported"):
        too_small.validate_image_size()


@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [
        (400, "openai_image_request_failed"),
        (401, "openai_image_authentication_failed"),
        (403, "openai_image_authentication_failed"),
        (404, "openai_image_model_unavailable"),
        (429, "openai_image_rate_limited"),
        (503, "openai_image_provider_unavailable"),
    ],
)
def test_openai_image_status_errors_are_classified(status_code, error_code):
    assert OpenAIImageGenerationTool._api_error_code(status_code) == error_code


@pytest.mark.asyncio
async def test_workspace_png_writer_rejects_invalid_png(monkeypatch):
    settings = SimpleNamespace(
        workspace_internal_token=SecretStr("internal-token"),
    )
    monkeypatch.setattr(
        image_generation_module, "get_settings", lambda: settings
    )

    async def before_workspace_write():
        raise AssertionError("invalid PNG must not reach the write check")

    with pytest.raises(RuntimeError, match="invalid_png"):
        await save_generated_png(
            uuid4(),
            "invalid.png",
            b"not a PNG",
            before_workspace_write,
        )


@pytest.mark.asyncio
async def test_openai_image_is_generated_without_retry_and_saved_to_workspace(
    monkeypatch,
):
    task_id = uuid4()
    calls = []
    saved_files = []
    client_options = {}
    settings = SimpleNamespace(
        agent_timeout_seconds=120,
        openai_base_url=None,
        workspace_internal_token=SecretStr("internal-token"),
        api_internal_base_url="http://api:8000",
    )
    monkeypatch.setattr(
        image_generation_module, "get_settings", lambda: settings
    )
    monkeypatch.setattr(
        image_generation_module,
        "read_openai_api_key",
        lambda: "test-key",
    )

    async def handle_request(request: httpx.Request):
        saved_files.append(request)
        return httpx.Response(200, json={"size_bytes": len(_PNG)})

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            client_options.update(kwargs)
            self.images = self

        async def generate(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(
                        b64_json=base64.b64encode(_PNG).decode()
                    )
                ]
            )

        async def close(self):
            return None

    monkeypatch.setattr(
        image_generation_module, "AsyncOpenAI", FakeOpenAIClient
    )

    async def before_workspace_write():
        assert calls

    tool = OpenAIImageGenerationTool(
        task_id=task_id,
        model="gpt-image-2.5-sunburst",
        before_workspace_write=before_workspace_write,
        transport=httpx.MockTransport(handle_request),
    )
    result = await tool._generate(
        OpenAIImageGenerationArguments(
            image_prompt={
                "positive_prompt": "A paper crane",
                "negative_prompt": "blurry",
            },
            filename="crane.png",
            width=1024,
            height=1024,
        )
    )

    assert calls == [
        {
            "model": "gpt-image-2.5-sunburst",
            "prompt": "A paper crane\n\nDo not include: blurry",
            "size": "1024x1024",
            "quality": "auto",
            "output_format": "png",
            "timeout": 120,
        }
    ]
    assert client_options["max_retries"] == 0
    assert result["artifact_path"].startswith("artifacts/crane-")
    assert result["width"] == 1024
    assert result["seed"] is None
    assert result["quality"] == "auto"
    assert result["quality"] == "auto"
    assert len(saved_files) == 1
    assert saved_files[0].headers["X-Workspace-Token"] == "internal-token"
    assert saved_files[0].url.params["role"] == "final"
    assert saved_files[0].content == _PNG
