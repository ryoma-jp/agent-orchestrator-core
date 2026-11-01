import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from app import agent_runtime as agent_runtime_module
from app import comfyui as comfyui_module
from app.agent_schemas import ChecklistItem, ChecklistReviewResult
from app.comfyui import ComfyUIImageArguments, ComfyUIImageGenerationTool
from app.quality import normalize_checklist_results
from app.tools import ToolRegistry

_PNG = b"\x89PNG\r\n\x1a\nimage-bytes"
_PROMPT_ID = "f5f4701a-1e4c-43e9-880d-1574b725e901"


@pytest.mark.asyncio
async def test_comfyui_image_is_validated_and_saved_to_task_workspace(
    monkeypatch,
):
    task_id = uuid4()
    requests = []
    saved_files = []
    workflow_prefix = {}
    settings = SimpleNamespace(
        comfyui_base_url="http://comfyui:8188",
        comfyui_timeout_seconds=10,
        comfyui_max_image_bytes=1_048_576,
        workspace_internal_token=SecretStr("internal-token"),
        api_internal_base_url="http://api:8000",
    )
    monkeypatch.setattr(comfyui_module, "get_settings", lambda: settings)

    async def handle_request(request: httpx.Request):
        requests.append(request)
        if request.url.path == "/prompt":
            payload = json.loads(request.content)
            workflow = payload["prompt"]
            assert workflow["1"]["class_type"] == "CheckpointLoaderSimple"
            assert workflow["1"]["inputs"]["ckpt_name"] == "model.safetensors"
            assert workflow["2"]["inputs"]["text"] == "A paper crane"
            workflow_prefix["value"] = workflow["7"]["inputs"][
                "filename_prefix"
            ]
            return httpx.Response(200, json={"prompt_id": _PROMPT_ID})
        if request.url.path == f"/history/{_PROMPT_ID}":
            return httpx.Response(
                200,
                json={
                    _PROMPT_ID: {
                        "status": {"status_str": "success"},
                        "outputs": {
                            "7": {
                                "images": [
                                    {
                                        "filename": (
                                            f"{workflow_prefix['value']}"
                                            "_00001_.png"
                                        ),
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                },
            )
        if request.url.path == "/view":
            return httpx.Response(
                200, content=_PNG, headers={"content-type": "image/png"}
            )
        if "/workspace/internal/artifacts/" in request.url.path:
            saved_files.append(request)
            return httpx.Response(200, json={"size_bytes": len(_PNG)})
        return httpx.Response(404)

    async def ensure_active():
        assert requests[-1].url.path == "/view"

    tool = ComfyUIImageGenerationTool(
        task_id=task_id,
        checkpoint_name="model.safetensors",
        before_workspace_write=ensure_active,
        transport=httpx.MockTransport(handle_request),
    )
    result = await tool._generate(
        ComfyUIImageArguments(
            prompt=" A paper crane ",
            negative_prompt="blurry",
            filename="crane.png",
            width=512,
            height=768,
            steps=20,
        )
    )

    assert result["artifact_path"].startswith("artifacts/crane-")
    assert result["artifact_path"].endswith(".png")
    assert len(saved_files) == 1
    saved_file = saved_files[0]
    assert saved_file.method == "PUT"
    assert saved_file.headers["X-Workspace-Token"] == "internal-token"
    assert saved_file.url.params["role"] == "final"
    assert saved_file.content == _PNG


def test_comfyui_tool_requires_explicit_write_permission():
    tool = ComfyUIImageGenerationTool(
        task_id=uuid4(),
        checkpoint_name="model.safetensors",
        before_workspace_write=lambda: None,
    )
    registration = tool.registration()
    default_registry = ToolRegistry([registration])
    enabled_registry = ToolRegistry(
        [registration], exposed_risk_classes={"read_only", "writes_data"}
    )

    assert default_registry.openai_definitions() == []
    assert enabled_registry.openai_definitions()[0]["name"] == (
        "comfyui_generate_image"
    )


@pytest.mark.parametrize(
    ("enabled", "image_provider", "include_image_tool", "expected"),
    [
        (True, "comfyui", True, True),
        (True, "comfyui", False, False),
        (True, "openai", True, False),
        (True, "ollama", True, False),
        (False, "comfyui", True, False),
    ],
)
def test_subtask_image_tool_follows_image_provider(
    monkeypatch, enabled, image_provider, include_image_tool, expected
):
    monkeypatch.setattr(
        agent_runtime_module,
        "get_settings",
        lambda: SimpleNamespace(comfyui_enabled=enabled),
    )

    async def before_workspace_write():
        return None

    registry = agent_runtime_module.subtask_tool_registry(
        uuid4(),
        {
            "worker_image_provider": image_provider,
            "worker_image_model": "model.safetensors",
        },
        before_workspace_write,
        include_image_tool=include_image_tool,
    )
    tool_names = {
        definition["name"] for definition in registry.openai_definitions()
    }

    assert ("comfyui_generate_image" in tool_names) is expected


def test_comfyui_arguments_reject_unsafe_paths_and_dimensions():
    with pytest.raises(ValidationError):
        ComfyUIImageArguments(prompt="image", filename="../image.png")
    with pytest.raises(ValidationError):
        ComfyUIImageArguments(prompt="image", width=513)


def test_generated_png_paths_satisfy_file_presence_check():
    checklist = [
        ChecklistItem(
            id="image",
            description="Save the requested image",
            verification_mode="artifact_present",
        )
    ]
    review = ChecklistReviewResult(results=[])

    results = normalize_checklist_results(
        checklist,
        review,
        {
            "answer": "The image is ready.",
            "artifacts": [],
            "generated_files": ["artifacts/paper-crane-1234abcd.png"],
            "sources": [],
        },
    )
    unsafe_results = normalize_checklist_results(
        checklist,
        review,
        {
            "answer": "The image is ready.",
            "artifacts": [],
            "generated_files": ["artifacts/../outside.png"],
            "sources": [],
        },
    )

    assert results[0]["passed"]
    assert results[0]["evidence"] == "artifacts/paper-crane-1234abcd.png"
    assert not unsafe_results[0]["passed"]
