import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import httpx
import httpx2
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError
from starlette.requests import Request

from app.agent_schemas import (
    ChecklistItem,
    ChecklistItemResult,
    ChecklistReviewResult,
    GeneratedArtifact,
    ImageGenerationRecord,
    IntegratedAnswer,
    PlannedSubtask,
    SubtaskResult,
    TaskPlan,
    TaskTitle,
)
from app.dependencies import get_task_user, require_task_csrf
from app.models import (
    ApiAccessKey,
    ConversationMessage,
    Event,
    ImageGenerationRevision,
    Job,
    QualityReview,
    Subtask,
    Task,
    TaskFile,
    User,
)
from app.openai_client import AgentExecutionError, StructuredModelClient
from app.quality import (
    failed_required_checklist_items,
    normalize_checklist_results,
    required_checklist_items_passed,
)
from app.routers.tasks import (
    add_message,
    edit_message,
    list_image_revisions,
)
from app.routers.workspace import list_task_files, upload_task_file
from app.schemas import (
    AdminSettingsUpdate,
    MessageCreate,
    TaskCreate,
    TaskView,
    ToolBudgetApprovalDecision,
    TaskFileView,
    TavilyKeyRequest,
)
from app.security import hash_password, token_digest, verify_password
from app.tool_budget_approval import resolve_tool_budget_approval
from app.tools import (
    TavilySearchArguments,
    TavilySearchTool,
    ToolRegistration,
    ToolRegistry,
)
from app import workspaces
from app import model_catalog, openai_client as openai_client_module
from app.config import Settings


def test_task_instruction_is_bounded_and_nonempty():
    assert (
        TaskCreate(instruction="research this").instruction == "research this"
    )


def test_task_view_includes_generated_title():
    now = datetime.now(timezone.utc)
    task = Task(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        instruction="Create a concise guide to accessible color palettes",
        title="Accessible color palette guide",
        title_generated=True,
        status="running",
        created_at=now,
        updated_at=now,
        result=None,
        error=None,
    )

    assert TaskView.model_validate(task).model_dump()["title"] == (
        "Accessible color palette guide"
    )


@pytest.mark.asyncio
async def test_image_revision_history_is_owner_scoped_and_ordered():
    owner_id = uuid.uuid4()
    task_id = uuid.uuid4()
    subtask_id = uuid.uuid4()
    review_id = uuid.uuid4()
    earlier = ImageGenerationRevision(
        id=uuid.uuid4(),
        task_id=task_id,
        subtask_id=subtask_id,
        plan_version=1,
        conversation_sequence=3,
        generation_no=1,
        trigger_quality_review_id=None,
        source_artifact_paths=[],
        artifact_path="artifacts/initial.png",
        prompt="initial pool prompt",
        negative_prompt="collage",
        openai_prompt=None,
        model_id="checkpoint.safetensors",
        width=768,
        height=1024,
        steps=25,
        seed=123,
        quality=None,
        unmet_items=[],
        is_backfilled=False,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    later = ImageGenerationRevision(
        id=uuid.uuid4(),
        task_id=task_id,
        subtask_id=subtask_id,
        plan_version=1,
        conversation_sequence=3,
        generation_no=2,
        trigger_quality_review_id=review_id,
        source_artifact_paths=["artifacts/initial.png"],
        artifact_path="artifacts/revised.png",
        prompt="three women playing in one pool",
        negative_prompt="face grid, collage",
        openai_prompt=(
            "three women playing in one pool\n\nDo not include: "
            "face grid, collage"
        ),
        model_id="checkpoint.safetensors",
        width=768,
        height=1024,
        steps=30,
        seed=456,
        quality=None,
        unmet_items=[
            {
                "item_id": "scene",
                "description": "Three women play in one pool.",
                "evidence": "The first image was a face grid.",
                "deficiency": "The image shows unrelated faces.",
            }
        ],
        is_backfilled=False,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    class FakeResult:
        def all(self):
            return [
                (earlier, "Generate requested image"),
                (later, "Generate requested image"),
            ]

    class FakeSession:
        def __init__(self, owns_task):
            self.owns_task = owns_task
            self.history_read = False

        async def scalar(self, _statement):
            return task_id if self.owns_task else None

        async def execute(self, _statement):
            self.history_read = True
            return FakeResult()

    user = SimpleNamespace(id=owner_id)
    own_session = FakeSession(True)
    history = await list_image_revisions(task_id, user, own_session)

    assert [item.generation_no for item in history] == [1, 2]
    assert history[1].trigger_quality_review_id == review_id
    assert history[1].unmet_items[0].deficiency == (
        "The image shows unrelated faces."
    )
    assert history[1].prompt == "three women playing in one pool"
    assert history[1].openai_prompt == later.openai_prompt
    assert history[1].seed == 456
    assert own_session.history_read

    other_session = FakeSession(False)
    with pytest.raises(HTTPException) as exc_info:
        await list_image_revisions(task_id, user, other_session)
    assert exc_info.value.status_code == 404
    assert not other_session.history_read


def test_empty_task_instruction_is_rejected():
    try:
        TaskCreate(instruction="")
    except ValueError:
        return
    raise AssertionError("empty instruction was accepted")


def test_workspace_paths_reject_absolute_and_parent_paths():
    for path in ("/etc/passwd", "../secret", "inputs/../../secret", "a\\b"):
        with pytest.raises(workspaces.InvalidWorkspacePath):
            workspaces.normalize_workspace_path(path)


def test_workspace_routes_are_registered():
    from app.main import app

    paths = set(app.openapi()["paths"])
    assert "/api/tasks/{task_id}/workspace/files" in paths
    assert "/api/tasks/{task_id}/workspace/files/{relative_path}" in paths
    assert "/api/tasks/{task_id}/workspace/internal/inputs" in paths
    assert (
        "/api/tasks/{task_id}/workspace/internal/artifacts/{filename}"
        in paths
    )
    assert "/api/tasks/{task_id}/workspace/internal/{relative_path}" in paths


@pytest.mark.asyncio
async def test_workspace_file_view_uses_filesystem_modification_time(
    monkeypatch,
):
    task_id = uuid.uuid4()
    user = User(
        id=uuid.uuid4(),
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )
    metadata_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    filesystem_time = datetime(2025, 6, 7, 8, 9, tzinfo=timezone.utc)
    task_file = TaskFile(
        id=uuid.uuid4(),
        task_id=task_id,
        workspace_rel_path="inputs/brief.txt",
        role="input",
        source_kind="user_upload",
        source_ref="brief.txt",
        size_bytes=5,
        created_at=metadata_time,
        updated_at=metadata_time,
    )

    class FakeSession:
        async def scalars(self, _statement):
            return [task_file]

    async def owned_task(_db, _task_id, _owner_id):
        return None

    monkeypatch.setattr("app.routers.workspace._owned_task", owned_task)
    monkeypatch.setattr(
        "app.routers.workspace.list_workspace_entries",
        lambda _task_id: [
            {
                "kind": "file",
                "path": "inputs/brief.txt",
                "size_bytes": 5,
                "updated_at": filesystem_time.timestamp(),
            },
            {"kind": "directory", "path": "inputs"},
        ],
    )

    result = await list_task_files(task_id, user, FakeSession())

    assert result[0].updated_at == filesystem_time
    assert result[1].kind == "directory"
    assert result[1].path == "inputs"


def test_workspace_listing_includes_empty_directories(monkeypatch, tmp_path):
    task_id = uuid.uuid4()
    settings = SimpleNamespace(workspace_root=str(tmp_path / "workspaces"))
    monkeypatch.setattr(workspaces, "get_settings", lambda: settings)
    workspace = workspaces.task_workspace(task_id, create=True)
    (workspace / "empty").mkdir()
    (workspace / "nested" / "empty").mkdir(parents=True)
    (workspace / "with-file" / "note.txt").parent.mkdir()
    (workspace / "with-file" / "note.txt").write_text("contents")

    entries = workspaces.list_workspace_entries(task_id)
    entries_by_path = {entry["path"]: entry for entry in entries}

    assert entries_by_path["inputs"]["kind"] == "directory"
    assert entries_by_path["artifacts"]["kind"] == "directory"
    assert entries_by_path["empty"]["kind"] == "directory"
    assert entries_by_path["nested"]["kind"] == "directory"
    assert entries_by_path["nested/empty"]["kind"] == "directory"
    assert entries_by_path["with-file"]["kind"] == "directory"
    assert entries_by_path["with-file/note.txt"]["kind"] == "file"


def test_task_file_view_serializes_workspace_path_as_path():
    task_file = TaskFile(
        id=uuid.uuid4(),
        task_id=uuid.uuid4(),
        workspace_rel_path="artifacts/final-answer.md",
        role="final",
        source_kind="agent",
        source_ref="task_result",
        size_bytes=12,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    file_view = TaskFileView.model_validate(task_file)

    assert file_view.model_dump()["path"] == "artifacts/final-answer.md"
    assert file_view.model_dump()["kind"] == "file"


def test_tool_budget_approval_state_and_decision_are_serializable():
    task_id = uuid.uuid4()
    subtask = Subtask(
        id=uuid.uuid4(),
        task_id=task_id,
        title="Research sources",
        instruction="Find sources",
        model_id="worker",
        tool_calls_used=20,
        tool_calls_limit=40,
    )
    task = Task(
        id=task_id,
        owner_id=uuid.uuid4(),
        instruction="Research",
        status="waiting_approval",
        pending_tool_approval={
            "request_id": str(uuid.uuid4()),
            "subtask_id": str(subtask.id),
            "additional_calls": 20,
        },
    )
    decision = ToolBudgetApprovalDecision(
        request_id=uuid.UUID(task.pending_tool_approval["request_id"]),
        approved=True,
    )

    assert subtask.tool_calls_limit == 40
    assert task.pending_tool_approval["additional_calls"] == 20
    assert decision.approved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "approved",
        "initial_limit",
        "current_limit",
        "expected_status",
        "expected_limit",
        "expected_verdict",
    ),
    [
        (True, 20, 20, "revising", 40, "revision_requested"),
        (False, 20, 20, "partially_accepted", 20, "partially_accepted"),
        (True, 40, 40, "revising", 80, "revision_requested"),
        (True, 20, 120, "revising", 140, "revision_requested"),
    ],
)
async def test_tool_budget_decision_resumes_or_accepts_partial(
    approved,
    initial_limit,
    current_limit,
    expected_status,
    expected_limit,
    expected_verdict,
):
    task_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    subtask_id = uuid.uuid4()
    request_id = uuid.uuid4()
    task = Task(
        id=task_id,
        owner_id=owner_id,
        instruction="Research sources",
        status="waiting_approval",
        config_snapshot={"max_tool_calls_per_subtask": initial_limit},
        pending_tool_approval={
            "request_id": str(request_id),
            "plan_version": 4,
            "subtask_id": str(subtask_id),
            "additional_calls": initial_limit,
            "reviewer_model_id": "manager",
        },
    )
    subtask = Subtask(
        id=subtask_id,
        task_id=task_id,
        title="Research sources",
        instruction="Find sources",
        status="waiting_approval",
        plan_version=4,
        model_id="worker",
        quality_revision_count=0,
        tool_calls_used=current_limit,
        tool_calls_limit=current_limit,
        result={"quality_review": [{"item_id": "source", "passed": False}]},
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, subtask]
            self.added = []

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

        async def refresh(self, _value):
            return None

    session = FakeSession()
    result = await resolve_tool_budget_approval(
        task_id,
        owner_id,
        ToolBudgetApprovalDecision(
            request_id=request_id,
            approved=approved,
        ),
        session,
    )

    assert result.status == "queued"
    assert task.pending_tool_approval is None
    assert subtask.status == expected_status
    assert subtask.tool_calls_limit == expected_limit
    assert subtask.tool_calls_used == current_limit
    assert any(
        isinstance(item, QualityReview)
        and item.verdict == expected_verdict
        for item in session.added
    )
    assert any(
        isinstance(item, Job) and item.status == "pending"
        for item in session.added
    )
    decision_event = next(
        item
        for item in session.added
        if isinstance(item, Event)
        and item.type == "task.tool_budget_approval_resolved"
    )
    assert decision_event.payload_json["approved"] is approved


@pytest.mark.asyncio
async def test_tool_budget_request_pauses_task_and_persists_partial_result(
    monkeypatch,
):
    import app.orchestrator as orchestrator_module
    import app.orchestration_state as state_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    subtask = Subtask(
        id=uuid.uuid4(),
        task_id=task_id,
        title="Research sources",
        instruction="Find sources",
        status="reviewing",
        plan_version=2,
        model_id="worker",
        tool_calls_used=40,
        tool_calls_limit=40,
    )
    task = Task(
        id=task_id,
        owner_id=uuid.uuid4(),
        instruction="Research",
        status="running",
        pending_tool_approval=None,
    )
    job = Job(
        id=job_id,
        task_id=task_id,
        status="leased",
        lease_owner="test-worker",
        lease_expires_at=datetime.now(timezone.utc),
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, 4, subtask]
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        async def get(self, model, _identity):
            return job if model is Job else None

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    async def assert_lease(*_args):
        return None

    session = FakeSession()
    monkeypatch.setattr(state_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(state_module, "assert_lease_in_session", assert_lease)
    partial_result = {
        "status": "partial",
        "summary": "Collected evidence before the limit.",
        "quality_review": [
            {
                "item_id": "sources",
                "description": "Support each claim",
                "required": True,
                "passed": False,
                "deficiency": "More sources are needed.",
                "requires_additional_tool_calls": True,
            }
        ],
    }

    requested = await orchestrator_module._pause_for_tool_budget_approval(
        task_id=task_id,
        job_id=job_id,
        lease_owner="test-worker",
        row=subtask,
        plan_version=2,
        conversation_sequence=4,
        subtask_result=partial_result,
        review_results=partial_result["quality_review"],
        reviewer_model_id="manager",
        additional_calls=20,
    )

    assert requested
    assert task.status == "waiting_approval"
    assert task.pending_tool_approval["additional_calls"] == 20
    assert task.pending_tool_approval["subtask_id"] == str(subtask.id)
    assert subtask.status == "waiting_approval"
    assert subtask.result == partial_result
    assert job.status == "completed"
    assert any(
        isinstance(item, Event)
        and item.type == "task.tool_budget_approval_requested"
        for item in session.added
    )


def test_workspace_paths_reject_symlinks(monkeypatch, tmp_path):
    task_id = uuid.uuid4()
    monkeypatch.setattr(
        workspaces,
        "get_settings",
        lambda: SimpleNamespace(workspace_root=str(tmp_path / "workspaces")),
    )
    root = workspaces.task_workspace(task_id, create=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (root / "escape").symlink_to(outside)

    with pytest.raises(workspaces.InvalidWorkspacePath):
        workspaces.resolve_workspace_file(task_id, "escape")


@pytest.mark.asyncio
async def test_workspace_stream_write_is_atomic_and_task_scoped(
    monkeypatch, tmp_path
):
    task_id = uuid.uuid4()
    monkeypatch.setattr(
        workspaces,
        "get_settings",
        lambda: SimpleNamespace(workspace_root=str(tmp_path / "workspaces")),
    )

    async def chunks():
        yield b"first "
        yield b"file"

    size = await workspaces.write_workspace_stream(
        task_id, "inputs/brief.txt", chunks()
    )

    assert size == 10
    assert workspaces.resolve_workspace_file(
        task_id, "inputs/brief.txt"
    ).read_bytes() == b"first file"
    assert not (tmp_path / "workspaces" / "brief.txt").exists()


def test_admin_quality_revision_limits_are_bounded():
    settings = AdminSettingsUpdate(
        manager_model="manager",
        worker_model="worker",
        max_concurrent_agents=4,
        max_agent_attempts=3,
        max_quality_revisions_per_subtask=2,
        max_quality_revisions_per_task=10,
    )
    assert settings.max_quality_revisions_per_subtask == 2
    assert settings.max_quality_revisions_per_task == 10
    assert settings.manager_model_provider == "openai"
    assert settings.worker_model_provider == "openai"
    assert settings.worker_image_provider == "openai"
    assert settings.worker_image_model == "gpt-image-2.5-sunburst"
    with pytest.raises(ValidationError):
        AdminSettingsUpdate(
            manager_model="manager",
            worker_model="worker",
            max_concurrent_agents=4,
            max_agent_attempts=3,
            max_quality_revisions_per_subtask=6,
            max_quality_revisions_per_task=10,
        )


def test_default_model_settings_use_openai_and_requested_models(monkeypatch):
    for name in (
        "MANAGER_MODEL_PROVIDER",
        "MANAGER_MODEL",
        "WORKER_MODEL_PROVIDER",
        "WORKER_MODEL",
        "WORKER_IMAGE_PROVIDER",
        "WORKER_IMAGE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.manager_model_provider == "openai"
    assert settings.manager_model == "gpt-6.1-sol"
    assert settings.worker_model_provider == "openai"
    assert settings.worker_model == "gpt-6-luna"
    assert settings.worker_image_provider == "openai"
    assert settings.worker_image_model == "gpt-image-2.5-sunburst"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "payload", "expected_url", "expected_headers", "expected"),
    [
        (
            "openai",
            {"data": [{"id": "model-b"}, {"id": "model-a"}]},
            "https://api.openai.com/v1/models",
            {"Authorization": "Bearer catalog-key"},
            ["model-a", "model-b"],
        ),
        (
            "ollama",
            {"models": [{"name": "local-b"}, {"name": "local-a"}]},
            "http://ollama:11434/api/tags",
            {},
            ["local-a", "local-b"],
        ),
        (
            "comfyui",
            {
                "CheckpointLoaderSimple": {
                    "input": {
                        "required": {
                            "ckpt_name": [
                                [
                                    "checkpoint-b.safetensors",
                                    "checkpoint-a.safetensors",
                                ],
                                {},
                            ]
                        }
                    }
                }
            },
            "http://comfyui:8188/object_info/CheckpointLoaderSimple",
            {},
            ["checkpoint-a.safetensors", "checkpoint-b.safetensors"],
        ),
    ],
)
async def test_model_catalog_uses_provider_endpoint(
    monkeypatch, provider, payload, expected_url, expected_headers, expected
):
    captured = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, headers):
            captured.append((url, headers))
            return FakeResponse()

    monkeypatch.setattr(
        model_catalog,
        "get_settings",
        lambda: SimpleNamespace(
            openai_base_url="https://api.openai.com/v1",
            comfyui_base_url="http://comfyui:8188",
        ),
    )
    monkeypatch.setattr(
        model_catalog,
        "read_openai_model_catalog_key",
        lambda: "catalog-key",
    )
    monkeypatch.setattr(
        model_catalog,
        "httpx",
        SimpleNamespace(AsyncClient=lambda **_kwargs: FakeClient()),
    )

    models = await model_catalog.list_models(provider)

    assert models == expected
    assert captured == [(expected_url, expected_headers)]


@pytest.mark.asyncio
async def test_ollama_provider_uses_chat_completions_without_vault_key(
    monkeypatch,
):
    captured = {}

    class FakeClient:
        async def close(self):
            captured["closed"] = True

    class FakeChatCompletionsModel:
        def __init__(self, model, openai_client):
            captured["model"] = model
            captured["adapter_client"] = openai_client

        async def get_response(self, **request):
            captured["request"] = request
            return "ollama-result"

    def create_client(**kwargs):
        captured["client_kwargs"] = kwargs
        return FakeClient()

    monkeypatch.setattr(
        openai_client_module,
        "get_settings",
        lambda: SimpleNamespace(
            agent_timeout_seconds=12,
        ),
    )
    monkeypatch.setattr(openai_client_module, "AsyncOpenAI", create_client)
    monkeypatch.setattr(
        openai_client_module,
        "OpenAIChatCompletionsModel",
        FakeChatCompletionsModel,
    )
    monkeypatch.setattr(
        openai_client_module,
        "read_openai_api_key",
        lambda: pytest.fail("Ollama must not read the OpenAI key"),
    )

    result = await openai_client_module.ConfiguredProviderModel(
        "qwen2.5", "ollama", 1, None
    ).get_response(tracing="disabled")

    assert result == "ollama-result"
    assert captured["model"] == "qwen2.5"
    assert captured["client_kwargs"]["api_key"] == "ollama"
    assert captured["client_kwargs"]["base_url"] == "http://ollama:11434/v1"
    assert (
        captured["request"]["tracing"]
        is openai_client_module.ModelTracing.DISABLED
    )
    assert captured["closed"]


@pytest.mark.asyncio
async def test_ollama_provider_runs_structured_output_through_sdk(monkeypatch):
    requests = []
    sdk_client = openai_client_module.AsyncOpenAI

    def respond(request):
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen2.5",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"answer":"Local response"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    def create_client(**kwargs):
        return sdk_client(
            **kwargs,
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(respond)
            ),
        )

    monkeypatch.setattr(
        openai_client_module,
        "get_settings",
        lambda: SimpleNamespace(
            max_agent_attempts=1,
            agent_timeout_seconds=12,
        ),
    )
    monkeypatch.setattr(openai_client_module, "AsyncOpenAI", create_client)
    monkeypatch.setattr(
        openai_client_module,
        "read_openai_api_key",
        lambda: pytest.fail("Ollama must not read the OpenAI key"),
    )

    result = await StructuredModelClient().generate(
        model="qwen2.5",
        provider="ollama",
        schema_name="small_output",
        schema=SmallOutput,
        system_prompt="Return a short answer.",
        input_text="Answer locally.",
        max_attempts=1,
    )

    assert result.answer == "Local response"
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer ollama"
    assert requests[0].read()


def test_tavily_api_key_request_rejects_empty_keys():
    with pytest.raises(ValidationError):
        TavilyKeyRequest(api_key="")


def test_write_tavily_api_key_uses_configured_vault_path(monkeypatch):
    import app.vault as vault_module

    writes = []

    class FakeKVv2:
        def create_or_update_secret(self, **kwargs):
            writes.append(kwargs)

    settings = SimpleNamespace(
        vault_api_role_id_file="api-role-id",
        vault_api_secret_id_file="api-secret-id",
        vault_kv_mount="secret",
        vault_tavily_key_path="custom/tavily",
    )
    client = SimpleNamespace(
        secrets=SimpleNamespace(kv=SimpleNamespace(v2=FakeKVv2()))
    )
    monkeypatch.setattr(vault_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        vault_module,
        "_authenticated_client",
        lambda role_id_file, secret_id_file: client,
    )

    vault_module.write_tavily_api_key("tavily-secret")

    assert writes == [
        {
            "mount_point": "secret",
            "path": "custom/tavily",
            "secret": {"api_key": "tavily-secret"},
        }
    ]


def test_message_content_is_bounded_and_nonempty():
    assert MessageCreate(content="follow-up").content == "follow-up"


def test_task_plan_accepts_a_valid_dependency_graph():
    plan = TaskPlan(
        title="Research project requirements",
        summary="Research and synthesize",
        final_checklist=[
            ChecklistItem(
                id="answer",
                description="Provide the requested answer",
            )
        ],
        subtasks=[
            PlannedSubtask(
                title="Research",
                instruction="Find facts",
                artifact_goal="Provide source-backed facts",
                acceptance_checklist=[
                    ChecklistItem(
                        id="sources",
                        description="Include relevant sources",
                        verification_mode="source_present",
                    )
                ],
                dependencies=[],
                skill_names=[],
            ),
            PlannedSubtask(
                title="Synthesis",
                instruction="Combine facts",
                artifact_goal="Synthesize research into an answer",
                acceptance_checklist=[
                    ChecklistItem(
                        id="synthesis",
                        description="Use the research findings",
                    )
                ],
                dependencies=[0],
                skill_names=[],
            ),
        ],
    )
    assert plan.subtasks[1].dependencies == [0]


def test_task_plan_rejects_dependency_cycles():
    with pytest.raises(ValidationError, match="contains a cycle"):
        TaskPlan(
            title="Invalid plan",
            summary="Invalid plan",
            final_checklist=[
                ChecklistItem(id="complete", description="Complete")
            ],
            subtasks=[
                PlannedSubtask(
                    title="A",
                    instruction="A",
                    artifact_goal="A",
                    acceptance_checklist=[
                        ChecklistItem(id="a", description="A")
                    ],
                    dependencies=[1],
                    skill_names=[],
                ),
                PlannedSubtask(
                    title="B",
                    instruction="B",
                    artifact_goal="B",
                    acceptance_checklist=[
                        ChecklistItem(id="b", description="B")
                    ],
                    dependencies=[0],
                    skill_names=[],
                ),
            ],
        )


def test_task_plan_rejects_duplicate_checklist_ids():
    with pytest.raises(
        ValidationError, match="checklist item ids must be unique"
    ):
        TaskPlan(
            title="Task purpose",
            summary="Plan",
            final_checklist=[
                ChecklistItem(id="answer", description="First"),
                ChecklistItem(id="answer", description="Duplicate"),
            ],
            subtasks=[
                PlannedSubtask(
                    title="Research",
                    instruction="Research",
                    artifact_goal="Evidence",
                    acceptance_checklist=[
                        ChecklistItem(id="source", description="Find source")
                    ],
                    dependencies=[],
                    skill_names=[],
                )
            ],
        )


def test_checklist_review_result_preserves_item_evidence():
    review = ChecklistReviewResult.model_validate(
        {
            "results": [
                {
                    "item_id": "sources",
                    "passed": False,
                    "evidence": "No source list was attached.",
                    "deficiency": "Add sources relevant to the claim.",
                }
            ]
        }
    )
    assert review.results[0].item_id == "sources"
    assert not review.results[0].passed


@pytest.mark.asyncio
async def test_subtask_quality_review_retries_with_only_failed_items(
    monkeypatch,
):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    subtask_id = uuid.uuid4()
    stored = SimpleNamespace(
        id=subtask_id,
        title="Research",
        status="running",
        result=None,
        error=None,
        quality_revision_count=0,
    )

    class FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return sum(
                isinstance(item, QualityReview)
                and item.verdict == "revision_requested"
                for item in self.added
            )

        async def get(self, model, _identity):
            return stored if model is Subtask else None

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    session = FakeSession()

    async def assert_lease(*_args):
        return None

    review_responses = [
        ChecklistReviewResult(
            results=[
                {
                    "item_id": "finding",
                    "passed": False,
                    "evidence": "The finding is too general.",
                    "deficiency": "Name the relevant version.",
                }
            ]
        ),
        ChecklistReviewResult(
            results=[
                {
                    "item_id": "finding",
                    "passed": True,
                    "evidence": "The finding names version 2.4.",
                    "deficiency": "",
                }
            ]
        ),
    ]

    reviewed_image_paths = []

    async def review_checklist(**kwargs):
        reviewed_image_paths.append(kwargs["image_paths"])
        return normalize_checklist_results(
            kwargs["checklist"],
            review_responses.pop(0),
            kwargs["artifact"],
        )

    revision_feedback = []
    revision_contexts = []

    async def revise(*_args, **kwargs):
        revision_feedback.append(kwargs["revision_feedback"])
        revision_contexts.append(kwargs["image_revision_context"])
        return SubtaskResult(
            status="succeeded",
            summary="Version 2.4 is supported.",
            findings=["Version 2.4 is supported."],
            sources=[],
            tool_runs=[],
            questions=[],
            error="",
        )

    monkeypatch.setattr(orchestrator_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )
    monkeypatch.setattr(
        orchestrator_module, "_review_checklist", review_checklist
    )
    monkeypatch.setattr(orchestrator_module, "_execute_subtask", revise)

    review_subtask = orchestrator_module._review_subtask_result
    result, status, questions = await review_subtask(
        row=SimpleNamespace(
            id=subtask_id,
            title="Research",
            instruction="Find details",
            artifact_goal="Find a version-specific fact",
            acceptance_checklist=[
                {
                    "id": "finding",
                    "description": "Name the relevant version",
                    "required": True,
                    "verification_mode": "model",
                }
            ],
            dependencies=[],
            model_id="test-worker",
            quality_revision_count=0,
        ),
        initial_result=SubtaskResult(
            status="succeeded",
            summary="A generic finding.",
            findings=["A generic finding."],
            sources=[],
            tool_runs=[
                {
                    "tool_name": "comfyui_generate_image",
                    "request_summary": "",
                    "status": "succeeded",
                    "source_count": 0,
                    "error_code": "",
                    "artifact_paths": ["artifacts/cat.png"],
                }
            ],
            questions=[],
            error="",
        ),
        plan=SimpleNamespace(summary="Research task"),
        results={},
        conversation=[],
        task_id=task_id,
        job_id=job_id,
        lease_owner="worker-test",
        plan_version=1,
        config={
            "manager_model": "test-manager",
            "max_agent_attempts": 1,
            "max_quality_revisions_per_subtask": 1,
            "max_quality_revisions_per_task": 1,
        },
        conversation_sequence=1,
    )

    assert status == "accepted"
    assert questions == []
    assert result["summary"] == "Version 2.4 is supported."
    assert reviewed_image_paths == [["artifacts/cat.png"], []]
    assert revision_feedback == [
        "- Name the relevant version: Name the relevant version."
    ]
    review_row = next(
        item
        for item in session.added
        if isinstance(item, QualityReview)
        and item.verdict == "revision_requested"
    )
    assert revision_contexts[0]["trigger_quality_review_id"] == str(
        review_row.id
    )
    assert revision_contexts[0]["source_artifact_paths"] == [
        "artifacts/cat.png"
    ]
    assert revision_contexts[0]["unmet_items"][0]["item_id"] == "finding"
    assert stored.status == "accepted"
    assert stored.quality_revision_count == 1
    verdicts = [
        item.verdict
        for item in session.added
        if isinstance(item, QualityReview)
    ]
    assert verdicts == ["revision_requested", "accepted"]


@pytest.mark.asyncio
async def test_subtask_quality_review_stops_at_revision_limit(monkeypatch):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    subtask_id = uuid.uuid4()
    stored = SimpleNamespace(
        id=subtask_id,
        title="Research",
        status="running",
        result=None,
        error=None,
        quality_revision_count=0,
    )

    class FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return 0

        async def get(self, model, _identity):
            return stored if model is Subtask else None

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    session = FakeSession()

    async def assert_lease(*_args):
        return None

    async def generate(**_kwargs):
        return ChecklistReviewResult(
            results=[
                {
                    "item_id": "finding",
                    "passed": False,
                    "evidence": "No supporting evidence.",
                    "deficiency": "Add a source-backed finding.",
                }
            ]
        )

    async def should_not_revise(*_args, **_kwargs):
        raise AssertionError("revision budget is exhausted")

    monkeypatch.setattr(orchestrator_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )
    monkeypatch.setattr(orchestrator_module.model_client, "generate", generate)
    monkeypatch.setattr(
        orchestrator_module, "_execute_subtask", should_not_revise
    )

    review_subtask = orchestrator_module._review_subtask_result
    result, status, questions = await review_subtask(
        row=SimpleNamespace(
            id=subtask_id,
            title="Research",
            instruction="Find facts",
            artifact_goal="Provide evidence",
            acceptance_checklist=[
                {
                    "id": "finding",
                    "description": "Provide a source-backed finding",
                    "required": True,
                    "verification_mode": "model",
                }
            ],
            dependencies=[],
            model_id="test-worker",
            quality_revision_count=0,
        ),
        initial_result=SubtaskResult(
            status="succeeded",
            summary="An unsupported finding.",
            findings=["An unsupported finding."],
            sources=[],
            tool_runs=[],
            questions=[],
            error="",
        ),
        plan=SimpleNamespace(summary="Research task"),
        results={},
        conversation=[],
        task_id=task_id,
        job_id=job_id,
        lease_owner="worker-test",
        plan_version=1,
        config={
            "manager_model": "test-manager",
            "max_agent_attempts": 1,
            "max_quality_revisions_per_subtask": 0,
            "max_quality_revisions_per_task": 0,
        },
        conversation_sequence=1,
    )

    assert status == "partially_accepted"
    assert questions == []
    assert result["quality_review"][0]["deficiency"] == (
        "Add a source-backed finding."
    )
    assert stored.status == "partially_accepted"
    assert [
        item.verdict
        for item in session.added
        if isinstance(item, QualityReview)
    ] == ["partially_accepted"]


def test_stored_plan_restores_subtask_order_and_dependencies():
    from app.planning import restore_plan_from_rows as _restore_plan_from_rows

    first_id = uuid.uuid4()
    second_id = uuid.uuid4()
    rows = [
        SimpleNamespace(
            id=first_id,
            title="Research",
            instruction="Find facts",
            artifact_goal="Find source-backed facts",
            acceptance_checklist=[
                {
                    "id": "sources",
                    "description": "Attach sources",
                    "required": True,
                    "verification_mode": "source_present",
                }
            ],
            dependencies=[],
            skill_names=[],
        ),
        SimpleNamespace(
            id=second_id,
            title="Synthesis",
            instruction="Combine results",
            artifact_goal="Answer the user",
            acceptance_checklist=[
                {
                    "id": "answer",
                    "description": "Answer the question",
                    "required": True,
                    "verification_mode": "model",
                }
            ],
            dependencies=[str(first_id)],
            skill_names=[],
        ),
    ]

    restored = _restore_plan_from_rows(
        SimpleNamespace(
            summary="Research and synthesize",
            final_checklist=[
                {
                    "id": "final",
                    "description": "Provide a useful answer",
                    "required": True,
                    "verification_mode": "model",
                }
            ],
        ),
        rows,
    )

    assert restored.subtasks[1].dependencies == [0]
    assert restored.subtasks[0].artifact_goal == "Find source-backed facts"


@pytest.mark.asyncio
async def test_first_plan_generates_and_persists_purpose_title(monkeypatch):
    import app.orchestrator as orchestrator_module
    import app.planning as planning_module

    task = Task(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        instruction="Build an accessible project dashboard",
        status="planning",
        config_snapshot={},
    )
    generated_plan = TaskPlan(
        title="Accessible project dashboard",
        summary="Build and verify the project dashboard.",
        final_checklist=[
            ChecklistItem(id="dashboard", description="Answer the request")
        ],
        subtasks=[
            PlannedSubtask(
                title="Review requirements",
                instruction="Identify dashboard requirements",
                artifact_goal="Summarize dashboard needs",
                acceptance_checklist=[
                    ChecklistItem(id="requirements", description="Summarize")
                ],
                dependencies=[],
                skill_names=["deep-research"],
            )
        ],
    )

    class FakeSession:
        def __init__(self, plan_version=None):
            self.scalar_values = [plan_version]
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        async def scalars(self, _statement):
            return []

        async def get(self, model, _identity):
            return task if model is Task else None

        def add(self, value):
            self.added.append(value)

        def add_all(self, values):
            self.added.extend(values)

        async def commit(self):
            return None

    sessions = [FakeSession(), FakeSession(plan_version=0)]
    generated_options = {}

    async def generate(**options):
        generated_options.update(options)
        return generated_plan

    async def assert_lease(*_args):
        return None

    async def assert_lease_in_session(*_args):
        return None

    async def format_input(*_args):
        return "user: Build an accessible project dashboard"

    async def no_new_messages(*_args):
        return False

    monkeypatch.setattr(
        planning_module, "SessionFactory", lambda: sessions.pop(0)
    )
    monkeypatch.setattr(planning_module, "assert_lease", assert_lease)
    monkeypatch.setattr(
        planning_module,
        "assert_lease_in_session",
        assert_lease_in_session,
    )
    monkeypatch.setattr(planning_module, "format_task_input", format_input)
    monkeypatch.setattr(
        planning_module,
        "list_private_skills",
        lambda: [
            {
                "name": "deep-research",
                "description": "Evidence-based research.",
            }
        ],
    )
    monkeypatch.setattr(
        planning_module, "requeue_if_new_messages", no_new_messages
    )
    monkeypatch.setattr(planning_module.model_client, "generate", generate)

    first_message = ConversationMessage(
        sequence=1,
        role="user",
        content="Build an accessible project dashboard",
    )
    result = await orchestrator_module._load_or_create_plan(
        task.id,
        uuid.uuid4(),
        "worker-test",
        [first_message],
        {
            "manager_model": "test-manager",
            "worker_model": "test-worker",
            "max_agent_attempts": 2,
            "max_agent_turns": 10,
        },
    )

    assert result is not None
    assert task.title == "Accessible project dashboard"
    assert task.title_generated is True
    assert "first instruction" in generated_options["system_prompt"]
    assert "dedicated image-generation subtask" in generated_options[
        "system_prompt"
    ]
    assert "Preserve each explicit visual requirement" in generated_options[
        "system_prompt"
    ]
    assert "deep-research" in generated_options["system_prompt"]
    assert result[2][0].skill_names == ["deep-research"]

    sessions.extend([FakeSession(), FakeSession(plan_version=1)])
    generated_plan.title = "Different follow-up title"
    followup = ConversationMessage(
        sequence=2,
        role="user",
        content="Also include deployment notes",
    )
    result = await orchestrator_module._load_or_create_plan(
        task.id,
        uuid.uuid4(),
        "worker-test",
        [first_message, followup],
        {
            "manager_model": "test-manager",
            "worker_model": "test-worker",
            "max_agent_attempts": 2,
            "max_agent_turns": 10,
        },
    )

    assert result is not None
    assert task.title == "Accessible project dashboard"


@pytest.mark.asyncio
async def test_restored_legacy_plan_generates_title_from_first_instruction(
    monkeypatch,
):
    import app.orchestrator as orchestrator_module
    import app.planning as planning_module

    task = Task(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        instruction="Create a concise guide to accessible color palettes",
        title="Create a concise guide to accessible color palettes",
        title_generated=False,
        status="planning",
        config_snapshot={},
    )
    record = SimpleNamespace(version=1)
    plan = TaskPlan(
        title="Accessible color palette guide",
        summary="Create a practical guide.",
        final_checklist=[
            ChecklistItem(id="guide", description="Provide the guide")
        ],
        subtasks=[
            PlannedSubtask(
                title="Review contrast needs",
                instruction="Review color contrast needs",
                artifact_goal="Summarize contrast needs",
                acceptance_checklist=[
                    ChecklistItem(id="contrast", description="Cover contrast")
                ],
                dependencies=[],
                skill_names=[],
            )
        ],
    )

    class FakeSession:
        def __init__(self, plan_record=None):
            self.plan_record = plan_record
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return self.plan_record

        async def scalars(self, _statement):
            return []

        async def get(self, model, _identity):
            return task if model is Task else None

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    sessions = [FakeSession(record), FakeSession()]
    generated_options = {}

    async def generate(**options):
        generated_options.update(options)
        return SimpleNamespace(title="Accessible color palette guide")

    async def assert_lease(*_args):
        return None

    async def assert_lease_in_session(*_args):
        return None

    monkeypatch.setattr(
        planning_module, "SessionFactory", lambda: sessions.pop(0)
    )
    monkeypatch.setattr(planning_module, "assert_lease", assert_lease)
    monkeypatch.setattr(
        planning_module,
        "assert_lease_in_session",
        assert_lease_in_session,
    )
    monkeypatch.setattr(
        planning_module,
        "restore_plan_from_rows",
        lambda *_args: plan,
    )
    monkeypatch.setattr(planning_module.model_client, "generate", generate)

    conversation = [
        ConversationMessage(
            sequence=1,
            role="user",
            content="Create a concise guide to accessible color palettes",
        ),
        ConversationMessage(
            sequence=2,
            role="assistant",
            content="I am preparing a plan.",
        ),
        ConversationMessage(
            sequence=3,
            role="user",
            content="Include a section on charts.",
        ),
    ]
    result = await orchestrator_module._load_or_create_plan(
        task.id,
        uuid.uuid4(),
        "worker-test",
        conversation,
        {
            "manager_model": "test-manager",
            "max_agent_attempts": 2,
            "max_agent_turns": 10,
        },
    )

    assert result is not None
    assert generated_options["schema"] is TaskTitle
    assert generated_options["input_text"] == conversation[0].content
    assert task.title == "Accessible color palette guide"
    assert task.title_generated is True


def test_deterministic_checklist_modes_override_model_review():
    checklist = [
        ChecklistItem(
            id="content",
            description="Include findings",
            verification_mode="non_empty",
        ),
        ChecklistItem(
            id="source",
            description="Attach a source",
            verification_mode="source_present",
        ),
        ChecklistItem(
            id="citation",
            description="Cite an attached source",
            verification_mode="markdown_citation",
        ),
    ]
    review = ChecklistReviewResult(results=[])
    artifact = {
        "summary": "A supported finding.",
        "findings": [],
        "sources": [{"url": "https://example.org/source"}],
    }

    results = normalize_checklist_results(
        checklist,
        review,
        {
            **artifact,
            "summary": (
                "A supported finding ([source](https://example.org/source))."
            ),
        },
    )

    assert required_checklist_items_passed(results)
    assert all(result["passed"] for result in results)


def test_requested_file_presence_and_content_are_checked():
    checklist = [
        ChecklistItem(
            id="file",
            description="Generate the requested file",
            verification_mode="artifact_present",
        ),
        ChecklistItem(
            id="citation",
            description="Cite an attached source in the file",
            verification_mode="markdown_citation",
        ),
    ]
    source_url = "https://example.org/source"
    review = ChecklistReviewResult(results=[])
    generated = {
        "answer": "Created the requested file.",
        "artifacts": [
            {
                "filename": "research.md",
                "content": f"# Findings\n\n[Source]({source_url})",
            }
        ],
        "sources": [{"url": source_url}],
    }

    results = normalize_checklist_results(checklist, review, generated)
    missing_results = normalize_checklist_results(
        checklist,
        review,
        {"answer": "I cannot attach files.", "artifacts": [], "sources": []},
    )

    assert required_checklist_items_passed(results)
    assert not required_checklist_items_passed(missing_results)


def test_missing_required_sources_requests_additional_tool_calls():
    results = normalize_checklist_results(
        [
            ChecklistItem(
                id="source",
                description="Cite a source",
                verification_mode="source_present",
            )
        ],
        ChecklistReviewResult(results=[]),
        {"answer": "An unsupported claim.", "sources": []},
    )

    assert not results[0]["passed"]
    assert results[0]["requires_additional_tool_calls"]


@pytest.mark.parametrize(
    "filename",
    ["../escape.md", "nested/file.md", "nested\\file.md", "bad\nname.md"],
)
def test_generated_artifact_rejects_unsafe_filenames(filename):
    with pytest.raises(ValidationError):
        GeneratedArtifact(filename=filename, content="Document body")


def test_missing_review_and_unsupported_citation_fail_closed():
    checklist = [
        ChecklistItem(id="support", description="Explain the claim"),
        ChecklistItem(
            id="citation",
            description="Link an attached source",
            verification_mode="markdown_citation",
        ),
    ]
    results = normalize_checklist_results(
        checklist,
        ChecklistReviewResult(results=[]),
        {
            "answer": "An unsupported claim ([source](https://evil.example)).",
            "sources": [{"url": "https://example.org/source"}],
        },
    )

    assert not required_checklist_items_passed(results)
    assert len(failed_required_checklist_items(results)) == 2


def test_review_preserves_additional_tool_requirement():
    results = normalize_checklist_results(
        [ChecklistItem(id="source", description="Find source")],
        ChecklistReviewResult(
            results=[
                ChecklistItemResult(
                    item_id="source",
                    passed=False,
                    evidence="",
                    deficiency="No source supports this claim.",
                    requires_additional_tool_calls=True,
                )
            ]
        ),
        {"answer": "A claim without source."},
    )

    assert results[0]["requires_additional_tool_calls"] is True


@pytest.mark.asyncio
async def test_partial_review_requests_approval_for_more_research(monkeypatch):
    import app.orchestrator as orchestrator_module

    row = SimpleNamespace(
        id=uuid.uuid4(),
        task_id=uuid.uuid4(),
        title="Research sources",
        status="running",
        plan_version=1,
        tool_calls_used=20,
        tool_calls_limit=20,
        quality_revision_count=0,
        acceptance_checklist=[
            {
                "id": "sources",
                "description": "Support the claims with sources",
                "required": True,
                "verification_mode": "model",
            }
        ],
        artifact_goal="Provide source-backed facts",
    )
    requested = []

    async def save_result(*_args, **_kwargs):
        return None

    async def review(*_args, **_kwargs):
        return [
            {
                "item_id": "sources",
                "description": "Support the claims with sources",
                "required": True,
                "passed": False,
                "evidence": "Current sources do not cover one claim.",
                "deficiency": "A primary source is needed.",
                "requires_additional_tool_calls": True,
            }
        ]

    async def pause_for_approval(**kwargs):
        requested.append(kwargs)
        return True

    async def quality_revision_count(_task_id):
        return 0

    monkeypatch.setattr(
        orchestrator_module, "_save_subtask_result", save_result
    )
    monkeypatch.setattr(orchestrator_module, "_review_checklist", review)
    monkeypatch.setattr(
        orchestrator_module,
        "_quality_revision_count",
        quality_revision_count,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_pause_for_tool_budget_approval",
        pause_for_approval,
    )
    config = {
        "manager_model": "manager",
        "max_agent_attempts": 2,
        "max_tool_calls_per_subtask": 40,
        "max_quality_revisions_per_subtask": 2,
        "max_quality_revisions_per_task": 10,
    }

    review_outcome = await orchestrator_module._review_subtask_result(
        row,
        SubtaskResult(
            status="partial",
            summary="Partial research",
            findings=["One supported claim."],
            sources=[],
            tool_runs=[],
            questions=[],
            error="tool_call_limit_exceeded",
        ),
        SimpleNamespace(summary="Research claims"),
        {},
        [],
        row.task_id,
        uuid.uuid4(),
        "test-worker",
        1,
        config,
        3,
    )
    result, status, questions = review_outcome

    assert status == "waiting_approval"
    assert questions == []
    assert result["quality_review"][0]["requires_additional_tool_calls"]
    assert requested[0]["additional_calls"] == 40


def test_task_plan_requires_nonblank_title():
    with pytest.raises(ValidationError, match="title must not be blank"):
        TaskPlan(
            title="  ",
            summary="Plan",
            final_checklist=[
                ChecklistItem(id="final", description="Answer the user")
            ],
            subtasks=[
                PlannedSubtask(
                    title="Research",
                    instruction="Find relevant information",
                    artifact_goal="Provide evidence",
                    acceptance_checklist=[
                        ChecklistItem(
                            id="evidence", description="Use evidence"
                        )
                    ],
                    dependencies=[],
                    skill_names=[],
                )
            ],
        )


def test_subtask_result_requires_questions_for_user_input():
    result = SubtaskResult(
        status="needs_input",
        summary="More information is needed.",
        findings=[],
        sources=[],
        tool_runs=[],
        questions=["Which audience should I focus on?"],
        error="",
    )
    assert result.questions == ["Which audience should I focus on?"]


def test_subtask_result_accepts_tool_budget_limited_partial_output():
    result = SubtaskResult(
        status="partial",
        summary="Search budget exhausted after collecting evidence.",
        findings=["A supported finding."],
        sources=[],
        tool_runs=[],
        questions=[],
        error="tool_call_limit_exceeded",
    )

    assert result.status == "partial"
    with pytest.raises(ValidationError, match="requires a question"):
        SubtaskResult(
            status="needs_input",
            summary="More information is needed.",
            findings=[],
            sources=[],
            tool_runs=[],
            questions=[],
            error="",
        )


@pytest.mark.asyncio
async def test_user_questions_pause_task_and_are_saved_as_assistant_message(
    monkeypatch,
):
    import app.orchestrator as orchestrator_module
    import app.orchestration_state as state_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    lease_owner = "worker-test"
    task = Task(
        id=task_id,
        owner_id=uuid.uuid4(),
        instruction="Research options",
        status="running",
        result="stale result",
        error={"code": "stale"},
    )
    job = Job(
        id=job_id,
        task_id=task_id,
        status="leased",
        lease_owner=lease_owner,
        lease_expires_at=None,
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, 1]
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        async def get(self, model, _identity):
            return job if model is Job else None

        def add(self, value):
            self.added.append(value)

        async def flush(self):
            return None

        async def commit(self):
            return None

    session = FakeSession()

    async def assert_lease(_session, _job_id, _lease_owner):
        return None

    monkeypatch.setattr(state_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(state_module, "assert_lease_in_session", assert_lease)

    await orchestrator_module._pause_for_user_input(
        task_id,
        job_id,
        lease_owner,
        1,
        [
            " Which audience should I focus on? ",
            "Which audience should I focus on?",
        ],
    )

    message = next(
        item for item in session.added if isinstance(item, ConversationMessage)
    )
    assert task.status == "needs_attention"
    assert task.result is None
    assert task.error is None
    assert message.role == "assistant"
    assert message.content == "1. Which audience should I focus on?"
    assert job.status == "completed"
    assert job.lease_owner is None
    assert any(
        isinstance(item, Event)
        and item.type == "task.status_changed"
        and item.payload_json["status"] == "needs_attention"
        and item.payload_json["conversation_sequence"] == 1
        for item in session.added
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_available", "artifact_requested", "generated_image"),
    [
        (True, True, True),
        (True, True, False),
        (False, True, False),
        (True, False, True),
        (True, False, False),
    ],
)
async def test_finish_task_persists_requested_artifact(
    monkeypatch, token_available, artifact_requested, generated_image
):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    task = Task(
        id=task_id,
        owner_id=uuid.uuid4(),
        instruction="Research options",
        status="running",
    )
    job = Job(
        id=job_id,
        task_id=task_id,
        status="leased",
        lease_owner="worker-test",
        lease_expires_at=None,
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, 2]
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        async def get(self, model, _identity):
            return job if model is Job else None

        def add(self, value):
            self.added.append(value)

        async def flush(self):
            return None

        async def commit(self):
            return None

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout
            self.requests = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def put(self, url, **kwargs):
            self.requests.append((url, kwargs))
            return FakeResponse()

    session = FakeSession()
    client = FakeClient(timeout=60.0)

    async def assert_lease(*_args):
        return None

    monkeypatch.setattr(orchestrator_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )
    monkeypatch.setattr(
        orchestrator_module,
        "get_settings",
        lambda: SimpleNamespace(
            workspace_internal_token=(
                SimpleNamespace(
                    get_secret_value=lambda: "test-workspace-token"
                )
                if token_available
                else None
            ),
            api_internal_base_url="http://api:8000",
            workspace_artifact_timeout_seconds=60,
            workspace_artifact_max_attempts=3,
        ),
    )
    monkeypatch.setattr(
        orchestrator_module.httpx,
        "AsyncClient",
        lambda timeout: client,
    )
    report = {
        "plan_version": 1,
        "unmet_items": [{"item_id": "sources", "description": "Cite sources"}],
    }

    finish = orchestrator_module._finish_task(
        task_id,
        job_id,
        "worker-test",
        IntegratedAnswer(
            answer="Partial answer",
            caveats=[],
            artifacts=(
                [
                    GeneratedArtifact(
                        filename="research.md",
                        content=(
                            "# Findings\n\n"
                            "The requested source-backed content."
                        ),
                    )
                ]
                if artifact_requested
                else []
            ),
        ),
        2,
        fully_completed=False,
        completion_report=report,
        generated_artifact_paths=(
            ["artifacts/generated-image-1234abcd.png"]
            if generated_image
            else []
        ),
    )
    if artifact_requested and not token_available:
        with pytest.raises(
            RuntimeError, match="^workspace_artifact_persistence_unavailable$"
        ):
            await finish
        assert task.status == "running"
        assert session.added == []
        assert client.requests == []
        return

    await finish

    assert task.status == "partially_completed"
    expected_paths = []
    if artifact_requested:
        expected_paths.append("artifacts/research.md")
    if generated_image:
        expected_paths.append("artifacts/generated-image-1234abcd.png")
    expected_answer = "Partial answer"
    if expected_paths:
        expected_answer += "\n\nSaved files: " + ", ".join(
            f"`{path}`" for path in expected_paths
        )
    assert task.result == expected_answer
    assert task.error is None
    assert task.completion_report == report
    assert job.status == "completed"
    assert any(
        isinstance(item, Event) and item.type == "task.partially_completed"
        for item in session.added
    )
    completion_event = next(
        item
        for item in session.added
        if isinstance(item, Event)
        and item.type == "task.partially_completed"
    )
    assert completion_event.payload_json["artifact_paths"] == expected_paths
    assert len(client.requests) == int(artifact_requested)
    if artifact_requested:
        assert client.timeout == 60
        assert client.requests[0][0].endswith(
            "/workspace/internal/artifacts/research.md"
        )
        assert client.requests[0][1]["content"] == (
            "# Findings\n\nThe requested source-backed content.".encode(
                "utf-8"
            )
        )
    assert not any(
        isinstance(item, Event) and item.type == "task.failed"
        for item in session.added
    )


@pytest.mark.asyncio
async def test_internal_workspace_write_uses_atomic_taskfile_upsert(
    monkeypatch,
):
    import app.routers.workspace as workspace_module
    from sqlalchemy.dialects import postgresql
    from starlette.requests import Request

    task_id = uuid.uuid4()
    task = SimpleNamespace(deletion_requested_at=None)
    stored_file = SimpleNamespace(path="artifacts/report.md", size_bytes=24)

    class FakeSession:
        def __init__(self):
            self.statements = []
            self.values = [task, stored_file]
            self.committed = False

        async def scalar(self, statement):
            self.statements.append(statement)
            return self.values.pop(0)

        async def commit(self):
            self.committed = True

    async def write_workspace_stream(_task_id, relative_path, _chunks):
        assert relative_path == "artifacts/report.md"
        return 24

    session = FakeSession()
    monkeypatch.setattr(
        workspace_module,
        "get_settings",
        lambda: SimpleNamespace(
            workspace_internal_token=SimpleNamespace(
                get_secret_value=lambda: "workspace-token"
            )
        ),
    )
    monkeypatch.setattr(
        workspace_module, "write_workspace_stream", write_workspace_stream
    )
    request = Request(
        {"type": "http", "method": "PUT", "path": "/", "headers": []}
    )

    result = await workspace_module.write_internal_task_file(
        task_id,
        "artifacts/report.md",
        request,
        x_workspace_token="workspace-token",
        db=session,
    )

    sql = str(session.statements[1].compile(dialect=postgresql.dialect()))
    assert result is stored_file
    assert session.committed
    assert "ON CONFLICT ON CONSTRAINT uq_task_files_path DO UPDATE" in sql


@pytest.mark.asyncio
async def test_internal_workspace_image_read_returns_generated_png(
    monkeypatch, tmp_path
):
    import app.routers.workspace as workspace_module

    task_id = uuid.uuid4()
    png_bytes = b"\x89PNG\r\n\x1a\nimage payload"
    image_path = tmp_path / "cat.png"
    image_path.write_bytes(png_bytes)
    task = SimpleNamespace(deletion_requested_at=None)
    task_file = SimpleNamespace(
        size_bytes=len(png_bytes), role="final", source_kind="agent"
    )

    class FakeSession:
        def __init__(self):
            self.values = [task, task_file]

        async def scalar(self, _statement):
            return self.values.pop(0)

    monkeypatch.setattr(
        workspace_module,
        "get_settings",
        lambda: SimpleNamespace(
            workspace_internal_token=SimpleNamespace(
                get_secret_value=lambda: "workspace-token"
            ),
            comfyui_max_image_bytes=1_000_000,
        ),
    )
    monkeypatch.setattr(
        workspace_module,
        "resolve_workspace_file",
        lambda _task_id, relative_path: (
            image_path
            if relative_path == "artifacts/cat.png"
            else pytest.fail("unexpected workspace path")
        ),
    )

    response = await workspace_module.read_internal_task_artifact(
        task_id,
        "cat.png",
        x_workspace_token="workspace-token",
        db=FakeSession(),
    )

    assert response.media_type == "image/png"
    assert response.body == png_bytes
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_internal_workspace_image_read_rejects_non_agent_file(
    monkeypatch,
):
    import app.routers.workspace as workspace_module

    task = SimpleNamespace(deletion_requested_at=None)
    task_file = SimpleNamespace(
        size_bytes=16, role="final", source_kind="user_edit"
    )

    class FakeSession:
        def __init__(self):
            self.values = [task, task_file]

        async def scalar(self, _statement):
            return self.values.pop(0)

    monkeypatch.setattr(
        workspace_module,
        "get_settings",
        lambda: SimpleNamespace(
            workspace_internal_token=SimpleNamespace(
                get_secret_value=lambda: "workspace-token"
            ),
            comfyui_max_image_bytes=1_000_000,
        ),
    )

    with pytest.raises(HTTPException) as error:
        await workspace_module.read_internal_task_artifact(
            uuid.uuid4(),
            "cat.png",
            x_workspace_token="workspace-token",
            db=FakeSession(),
        )
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_internal_workspace_image_read_returns_413_above_image_limit(
    monkeypatch,
):
    import app.routers.workspace as workspace_module

    task = SimpleNamespace(deletion_requested_at=None)
    task_file = SimpleNamespace(
        size_bytes=workspace_module.MAX_REVIEW_IMAGE_BYTES + 1,
        role="final",
        source_kind="agent",
    )

    class FakeSession:
        def __init__(self):
            self.values = [task, task_file]

        async def scalar(self, _statement):
            return self.values.pop(0)

    monkeypatch.setattr(
        workspace_module,
        "get_settings",
        lambda: SimpleNamespace(
            workspace_internal_token=SimpleNamespace(
                get_secret_value=lambda: "workspace-token"
            ),
            comfyui_max_image_bytes=workspace_module.MAX_REVIEW_IMAGE_BYTES,
        ),
    )

    with pytest.raises(HTTPException) as error:
        await workspace_module.read_internal_task_artifact(
            uuid.uuid4(),
            "large.png",
            x_workspace_token="workspace-token",
            db=FakeSession(),
        )
    assert error.value.status_code == 413


@pytest.mark.asyncio
async def test_review_image_loader_fetches_only_generated_artifact_paths(
    monkeypatch,
):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    png_bytes = b"\x89PNG\r\n\x1a\nimage payload"
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "image/png"}, content=png_bytes
        )

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        orchestrator_module.httpx,
        "AsyncClient",
        lambda **options: real_async_client(
            transport=transport, **options
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "get_settings",
        lambda: SimpleNamespace(
            api_internal_base_url="http://api",
            workspace_internal_token=SimpleNamespace(
                get_secret_value=lambda: "workspace-token"
            ),
            workspace_artifact_timeout_seconds=5,
        ),
    )

    selected_paths, images, omitted_paths = (
        await orchestrator_module._load_review_images(
            task_id,
            [
                "artifacts/cat.png",
                "artifacts/../outside.png",
                "inputs/secret.png",
            ],
        )
    )

    assert selected_paths == ["artifacts/cat.png"]
    assert images == [png_bytes]
    assert omitted_paths == []
    assert len(requests) == 1
    assert requests[0].url.path.endswith(
        f"/api/tasks/{task_id}/workspace/internal/artifacts/cat.png"
    )
    assert requests[0].headers["x-workspace-token"] == "workspace-token"


@pytest.mark.asyncio
async def test_checklist_review_passes_loaded_images_to_model(monkeypatch):
    from io import BytesIO

    import app.orchestrator as orchestrator_module
    from PIL import Image

    task_id = uuid.uuid4()
    png_buffer = BytesIO()
    Image.new("RGB", (256, 256), "white").save(
        png_buffer, format="PNG"
    )
    png_bytes = png_buffer.getvalue()
    loaded_paths = []
    model_images = []
    model_inputs = []
    model_system_prompts = []

    async def load_images(_task_id, image_paths):
        loaded_paths.extend(image_paths)
        return image_paths, [png_bytes], ["artifacts/omitted.png"]

    async def generate(**kwargs):
        model_images.extend(kwargs["input_images"])
        model_inputs.append(
            orchestrator_module.json.loads(kwargs["input_text"])
        )
        model_system_prompts.append(kwargs["system_prompt"])
        return ChecklistReviewResult(
            results=[
                {
                    "item_id": "image-content",
                    "passed": False,
                    "evidence": "",
                    "deficiency": "The image shows the wrong scene.",
                    "revision_target": "image",
                    "target_image_paths": [
                        "artifacts/cat.png",
                        "artifacts/not-attached.png",
                    ],
                },
                {
                        "item_id": "system-image-visual-integrity",
                        "passed": False,
                        "evidence": "The hand near the cup is visible.",
                        "deficiency": "The fingers do not connect naturally.",
                        "revision_target": "image",
                        "target_image_paths": ["artifacts/cat.png"],
                    },
            ]
        )

    monkeypatch.setattr(
        orchestrator_module, "_load_review_images", load_images
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_private_skill_if_present",
        lambda name: (
            "Inspect anatomy and background details."
            if name == "image-generation"
            else None
        ),
    )
    monkeypatch.setattr(orchestrator_module.model_client, "generate", generate)

    results = await orchestrator_module._review_checklist(
        model="test-manager",
        task_id=task_id,
        image_paths=["artifacts/cat.png"],
        checklist=[
            ChecklistItem(
                id="image-content",
                description="The image shows a cat.",
            )
        ],
        artifact={"generated_files": ["artifacts/cat.png"]},
        plan_summary="Generate a cat image",
        instruction="Draw a cat",
        max_attempts=1,
    )

    assert loaded_paths == ["artifacts/cat.png"]
    assert model_images[0] == png_bytes
    assert len(model_images) == 5
    assert model_inputs[0]["image_review"] == {
        "attached_paths": ["artifacts/cat.png"],
        "omitted_paths": ["artifacts/omitted.png"],
        "image_views": [
            {"artifact_path": "artifacts/cat.png", "view": "full image"},
            {
                "artifact_path": "artifacts/cat.png",
                "view": "top-left detail crop",
            },
            {
                "artifact_path": "artifacts/cat.png",
                "view": "top-right detail crop",
            },
            {
                "artifact_path": "artifacts/cat.png",
                "view": "bottom-left detail crop",
            },
            {
                "artifact_path": "artifacts/cat.png",
                "view": "bottom-right detail crop",
            },
        ],
        "omitted_detail_views": [],
        "image_views_follow_input_image_order": True,
    }
    assert "Inspect anatomy and background details." in (
        model_system_prompts[0]
    )
    assert "ChecklistReviewResult schema" in model_system_prompts[0]
    assert not results[0]["passed"]
    assert results[0]["revision_target"] == "image"
    assert results[0]["target_image_paths"] == ["artifacts/cat.png"]
    visual_integrity = next(
        result
        for result in results
        if result["item_id"].startswith("system-image-visual-integrity")
    )
    assert not visual_integrity["passed"]
    assert visual_integrity["revision_target"] == "image"
    assert visual_integrity["target_image_paths"] == ["artifacts/cat.png"]


@pytest.mark.asyncio
async def test_review_image_loader_randomly_caps_image_count(monkeypatch):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    png_bytes = b"\x89PNG\r\n\x1a\nimage"
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "image/png"}, content=png_bytes
        )

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        orchestrator_module.httpx,
        "AsyncClient",
        lambda **options: real_async_client(
            transport=transport, **options
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "get_settings",
        lambda: SimpleNamespace(
            api_internal_base_url="http://api",
            workspace_internal_token=SimpleNamespace(
                get_secret_value=lambda: "workspace-token"
            ),
            workspace_artifact_timeout_seconds=5,
        ),
    )
    monkeypatch.setattr(
        orchestrator_module.random, "shuffle", lambda paths: paths.reverse()
    )
    candidate_paths = [
        f"artifacts/image-{index}.png" for index in range(7)
    ]

    selected_paths, images, omitted_paths = (
        await orchestrator_module._load_review_images(
            task_id, candidate_paths
        )
    )

    assert selected_paths == list(reversed(candidate_paths[2:]))
    assert images == [png_bytes] * 5
    assert omitted_paths == list(reversed(candidate_paths[:2]))
    assert len(requests) == 5


@pytest.mark.asyncio
async def test_review_image_loader_skips_images_that_do_not_fit_byte_budget(
    monkeypatch,
):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    images_by_name = {
        "first.png": b"\x89PNG\r\n\x1a\n1234",
        "too-large-for-remaining.png": b"\x89PNG\r\n\x1a\n1234",
        "last.png": b"\x89PNG\r\n\x1a\n",
    }
    requests = []

    def respond(request):
        requests.append(request)
        filename = request.url.path.rsplit("/", 1)[-1]
        image = images_by_name[filename]
        return httpx.Response(
            200,
            headers={
                "content-type": "image/png",
                "content-length": str(len(image)),
            },
            content=image,
        )

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        orchestrator_module.httpx,
        "AsyncClient",
        lambda **options: real_async_client(
            transport=transport, **options
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "get_settings",
        lambda: SimpleNamespace(
            api_internal_base_url="http://api",
            workspace_internal_token=SimpleNamespace(
                get_secret_value=lambda: "workspace-token"
            ),
            workspace_artifact_timeout_seconds=5,
        ),
    )
    monkeypatch.setattr(
        orchestrator_module, "MAX_REVIEW_IMAGE_BYTES", 20
    )
    monkeypatch.setattr(
        orchestrator_module.random, "shuffle", lambda _paths: None
    )

    selected_paths, selected_images, omitted_paths = (
        await orchestrator_module._load_review_images(
            task_id,
            [
                "artifacts/first.png",
                "artifacts/too-large-for-remaining.png",
                "artifacts/last.png",
            ],
        )
    )

    assert selected_paths == ["artifacts/first.png", "artifacts/last.png"]
    assert selected_images == [
        images_by_name["first.png"],
        images_by_name["last.png"],
    ]
    assert omitted_paths == ["artifacts/too-large-for-remaining.png"]
    assert len(requests) == 3


@pytest.mark.asyncio
async def test_review_image_loader_skips_per_image_413_and_continues(
    monkeypatch,
):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    png_bytes = b"\x89PNG\r\n\x1a\nreviewable"
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path.endswith("/oversized.png"):
            return httpx.Response(413)
        return httpx.Response(
            200, headers={"content-type": "image/png"}, content=png_bytes
        )

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        orchestrator_module.httpx,
        "AsyncClient",
        lambda **options: real_async_client(
            transport=transport, **options
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "get_settings",
        lambda: SimpleNamespace(
            api_internal_base_url="http://api",
            workspace_internal_token=SimpleNamespace(
                get_secret_value=lambda: "workspace-token"
            ),
            workspace_artifact_timeout_seconds=5,
        ),
    )
    monkeypatch.setattr(
        orchestrator_module.random, "shuffle", lambda _paths: None
    )

    selected_paths, images, omitted_paths = (
        await orchestrator_module._load_review_images(
            task_id,
            ["artifacts/oversized.png", "artifacts/cat.png"],
        )
    )

    assert selected_paths == ["artifacts/cat.png"]
    assert images == [png_bytes]
    assert omitted_paths == ["artifacts/oversized.png"]
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_workspace_artifact_put_retries_read_timeout(monkeypatch):
    import httpx

    import app.orchestrator as orchestrator_module

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self):
            self.requests = []

        async def put(self, url, **kwargs):
            self.requests.append((url, kwargs))
            if len(self.requests) == 1:
                raise httpx.ReadTimeout("response timed out")
            return FakeResponse()

    async def no_wait(_delay):
        return None

    client = FakeClient()
    monkeypatch.setattr(orchestrator_module.asyncio, "sleep", no_wait)

    await orchestrator_module._put_workspace_artifact(
        client,
        "http://api/internal/artifacts/report.md",
        {"X-Workspace-Token": "test-token"},
        b"complete report",
        3,
    )

    assert len(client.requests) == 2
    assert client.requests[0] == client.requests[1]


@pytest.mark.asyncio
async def test_workspace_artifact_put_timeout_exhaustion_is_sanitized(
    monkeypatch,
):
    import httpx

    import app.orchestrator as orchestrator_module

    class FakeClient:
        def __init__(self):
            self.request_count = 0

        async def put(self, _url, **_kwargs):
            self.request_count += 1
            raise httpx.ReadTimeout("secret transport detail")

    async def no_wait(_delay):
        return None

    client = FakeClient()
    monkeypatch.setattr(orchestrator_module.asyncio, "sleep", no_wait)

    with pytest.raises(
        AgentExecutionError, match="^workspace_artifact_timeout$"
    ):
        await orchestrator_module._put_workspace_artifact(
            client,
            "http://api/internal/artifacts/report.md",
            {"X-Workspace-Token": "test-token"},
            b"complete report",
            3,
        )

    assert client.request_count == 3


@pytest.mark.asyncio
async def test_subtask_receives_conversation_history(monkeypatch):
    import json

    import app.orchestrator as orchestrator_module

    messages = [
        ConversationMessage(role="assistant", content="Which industry?"),
        ConversationMessage(role="user", content="IT, embedded software."),
    ]
    captured = {}

    async def assert_lease(*_args):
        return None

    async def reserve_tool(*_args):
        return None

    activities = []

    async def record_activity(*args, **kwargs):
        activities.append((args, kwargs))

    async def generate(**kwargs):
        captured.update(kwargs)
        await kwargs["before_tool_call"]("tavily_web_search")
        await kwargs["on_tool_result"](
            "tavily_web_search",
            '{"query":"public query","max_results":1}',
            {
                "retrieved_at": "2026-10-12T00:00:00+00:00",
                "results": [
                    {
                        "title": "Verified source",
                        "url": "https://example.org/source",
                        "content": "S" * 10_000,
                        "score": 0.8,
                        "published_date": "2026-10-01",
                    }
                ],
            },
        )
        return SubtaskResult(
            status="succeeded",
            summary="Completed",
            findings=[],
            sources=[],
            tool_runs=[],
            questions=[],
            error="",
        )

    monkeypatch.setattr(orchestrator_module, "_assert_lease", assert_lease)
    monkeypatch.setattr(
        orchestrator_module, "_reserve_subtask_tool_call", reserve_tool
    )
    monkeypatch.setattr(
        orchestrator_module, "_record_subtask_activity", record_activity
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_agent_options",
        lambda *_args, **_kwargs: {"before_model_call": assert_lease},
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_tavily_search_skill",
        lambda: "Skill-specific CLI search instructions.",
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_private_skill",
        lambda _name: "Private research skill instructions.",
    )
    monkeypatch.setattr(orchestrator_module.model_client, "generate", generate)

    result = await orchestrator_module._execute_subtask(
        SimpleNamespace(
            id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            plan_version=1,
            tool_calls_used=0,
            tool_calls_limit=20,
            title="Clarify profile",
            instruction="Proceed",
            artifact_goal=(
                "Summarize the user profile using verified evidence."
            ),
            acceptance_checklist=[
                {
                    "id": "profile",
                    "description": "Mention the stated industry",
                    "required": True,
                    "verification_mode": "model",
                }
            ],
            dependencies=[],
            model_id="test-worker",
            skill_names=["deep-research"],
        ),
        SimpleNamespace(summary="Plan"),
        {},
        messages,
        uuid.uuid4(),
        "worker-test",
        1,
    )

    prompt = json.loads(captured["input_text"])
    assert prompt["conversation"] == [
        {"role": "assistant", "content": "Which industry?"},
        {"role": "user", "content": "IT, embedded software."},
    ]
    assert prompt["artifact_goal"] == (
        "Summarize the user profile using verified evidence."
    )
    assert prompt["acceptance_checklist"][0]["id"] == "profile"
    assert result.sources[0].provider == "tavily"
    assert result.sources[0].url == "https://example.org/source"
    assert result.sources[0].score == 0.8
    assert len(result.sources[0].content) == 10_000
    assert result.tool_runs[0].request_summary == "public query"
    assert result.tool_runs[0].source_count == 1
    assert "do not ask again" in captured["system_prompt"]
    assert "tavily_web_search" in captured["system_prompt"]
    assert "Skill-specific CLI search instructions." in captured[
        "system_prompt"
    ]
    assert "Private research skill instructions." in captured[
        "system_prompt"
    ]
    activity_details = [
        (
            item[1]["activity"],
            item[1]["phase"],
            item[1].get("skill_name"),
            item[1].get("skill_source"),
        )
        for item in activities
    ]
    assert activity_details == [
        ("skill", "loaded", "tavily-search", "public"),
        ("skill", "fallback", "image-generation", "fallback"),
        ("skill", "loaded", "deep-research", "private"),
        ("tool", "completed", None, None),
    ]
    assert "Do not use tools" not in captured["system_prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("image_prompt_skill", "expected_skill_phase", "expected_skill_source"),
    [
        (
            "Use the installed image prompt skill.",
            "loaded",
            "private",
        ),
        (None, "fallback", "fallback"),
    ],
)
async def test_subtask_captures_image_prompt_for_revision(
    monkeypatch,
    image_prompt_skill,
    expected_skill_phase,
    expected_skill_source,
):
    import json

    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    previous = ImageGenerationRecord(
        artifact_path="artifacts/previous.png",
        prompt="three women at a pool",
        negative_prompt="collage, multiple panels",
        width=768,
        height=1024,
        steps=25,
    )
    new_arguments = {
        "image_prompt": {
            "positive_prompt": (
                "three adult women playing in one swimming pool scene"
            ),
            "negative_prompt": "collage, multiple panels, extra faces",
        },
        "filename": "pool.png",
        "width": 768,
        "height": 1024,
    }
    captured = {}
    checkpoints = []
    activities = []

    async def assert_lease(*_args):
        return None

    async def record_activity(*_args, **kwargs):
        activities.append(kwargs)

    async def checkpoint_image_generations(*args):
        checkpoints.append(args)

    async def generate(**kwargs):
        captured.update(kwargs)
        await kwargs["on_tool_result"](
            "openai_generate_image",
            json.dumps(new_arguments),
            {
                "artifact_path": "artifacts/revised-pool.png",
                "seed": None,
                "quality": "auto",
            },
        )
        return SubtaskResult(
            status="succeeded",
            summary="Generated a revised pool image.",
            findings=[],
            sources=[],
            tool_runs=[],
            questions=[],
            error="",
        )

    monkeypatch.setattr(
        orchestrator_module, "_assert_lease", assert_lease
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_agent_options",
        lambda *_args, **_kwargs: {"before_model_call": assert_lease},
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_record_subtask_activity",
        record_activity,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_checkpoint_image_generations",
        checkpoint_image_generations,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "subtask_tool_registry",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_tavily_search_skill",
        lambda: "Search skill instructions.",
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_private_skill_if_present",
        lambda _name: image_prompt_skill,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "get_settings",
        lambda: SimpleNamespace(comfyui_enabled=True),
    )
    monkeypatch.setattr(
        orchestrator_module.model_client, "generate", generate
    )

    result = await orchestrator_module._execute_subtask(
        SimpleNamespace(
            id=uuid.uuid4(),
            task_id=task_id,
            plan_version=1,
            tool_calls_used=0,
            tool_calls_limit=20,
            title="Generate an image",
            instruction="Draw three women playing in a pool.",
            artifact_goal="Show three women playing in one pool scene.",
            acceptance_checklist=[],
            dependencies=[],
            model_id="test-worker",
            skill_names=["image-generation-prompt"],
        ),
        SimpleNamespace(summary="Image task"),
        {},
        [],
        uuid.uuid4(),
        "worker-test",
        1,
        revision_feedback="Show exactly three people in one pool scene.",
        previous_image_generations=[previous],
        config={
            "max_tool_calls_per_subtask": 20,
            "worker_image_provider": "openai",
            "worker_image_model": "gpt-image-2.5-sunburst",
        },
    )

    runner_input = json.loads(captured["input_text"])
    saved = orchestrator_module._serialize_subtask_result(result)
    assert runner_input["previous_image_generations"] == [
        previous.model_dump(mode="json")
    ]
    assert runner_input["revision_feedback"] == (
        "Show exactly three people in one pool scene."
    )
    assert "image_prompt" in captured["system_prompt"]
    if image_prompt_skill is None:
        assert "Apply its prompt rules yourself" in captured["system_prompt"]
    else:
        assert image_prompt_skill in captured["system_prompt"]
    image_skill_activity = next(
        activity
        for activity in activities
        if activity.get("skill_name") == "image-generation"
    )
    assert image_skill_activity["phase"] == expected_skill_phase
    assert image_skill_activity["skill_source"] == expected_skill_source
    assert saved["image_generations"] == [
        {
            "artifact_path": "artifacts/revised-pool.png",
            "prompt": new_arguments["image_prompt"]["positive_prompt"],
            "negative_prompt": new_arguments["image_prompt"][
                "negative_prompt"
            ],
            "width": 768,
            "height": 1024,
            "steps": None,
            "seed": None,
            "quality": "auto",
            "openai_prompt": (
                "three adult women playing in one swimming pool scene\n\n"
                "Do not include: collage, multiple panels, extra faces"
            ),
        }
    ]
    assert "prompt" not in result.tool_runs[0].model_dump()
    assert len(checkpoints) == 1
    assert checkpoints[0][3][0].artifact_path == (
        "artifacts/revised-pool.png"
    )


def test_answer_revision_limit_keeps_configured_limit():
    import app.orchestrator as orchestrator_module

    answer_failure = [{"revision_target": "answer"}]
    assert orchestrator_module._image_revision_limit(5, answer_failure) == 5


def test_image_revision_limit_is_capped_and_preserves_lower_limits():
    import app.orchestrator as orchestrator_module

    image_failure = [{"revision_target": "image"}]
    answer_failure = [{"revision_target": "answer"}]
    assert orchestrator_module._image_revision_limit(0, image_failure) == 0
    assert orchestrator_module._image_revision_limit(1, image_failure) == 1
    assert orchestrator_module._image_revision_limit(5, image_failure) == 2
    assert orchestrator_module._image_revision_limit(5, answer_failure) == 5


def test_final_image_failure_maps_to_unique_generation_source():
    import app.orchestrator as orchestrator_module

    source_id = str(uuid.uuid4())
    generated = {
        "status": "succeeded",
        "image_generations": [
            {
                "artifact_path": "artifacts/pool.png",
                "prompt": "three people playing in a pool",
                "negative_prompt": "collage",
                "width": 768,
                "height": 1024,
                "steps": 25,
            }
        ],
    }
    failed_item = {
        "item_id": "scene",
        "required": True,
        "passed": False,
        "description": "Three women play in one pool scene.",
        "deficiency": "The image contains many separate faces.",
        "revision_target": "image",
        "target_image_paths": ["artifacts/pool.png"],
    }

    requests, unresolved = orchestrator_module._image_revision_requests(
        [failed_item], {source_id: generated}
    )

    assert requests == {source_id: [failed_item]}
    assert not unresolved
    assert orchestrator_module._stored_revision_feedback(
        {
            "image_revision_feedback": "Remove the unrelated face grid.",
            "quality_review": [],
        }
    ) == "Remove the unrelated face grid."

    failed_item["target_image_paths"] = ["artifacts/unattached.png"]
    requests, unresolved = orchestrator_module._image_revision_requests(
        [failed_item], {source_id: generated}
    )
    assert requests == {}
    assert unresolved


@pytest.mark.asyncio
async def test_begin_image_revision_persists_feedback_and_counter(monkeypatch):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    subtask_id = uuid.uuid4()
    previous_result = {"status": "succeeded", "summary": "Old image"}
    stored = SimpleNamespace(
        status="accepted",
        quality_revision_count=1,
        result=previous_result,
    )
    revision_context = {
        "trigger_quality_review_id": str(uuid.uuid4()),
        "source_artifact_paths": ["artifacts/old.png"],
        "unmet_items": [],
    }

    class FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _model, _identity, **_kwargs):
            return stored

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    async def assert_lease(*_args):
        return None

    row = SimpleNamespace(
        id=subtask_id,
        title="Generate image",
        status="accepted",
        quality_revision_count=1,
        result=previous_result,
    )
    session = FakeSession()
    monkeypatch.setattr(
        orchestrator_module, "SessionFactory", lambda: session
    )
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )

    await orchestrator_module._begin_image_revision(
        task_id,
        uuid.uuid4(),
        "worker-test",
        row,
        previous_result,
        "Remove unrelated faces and keep one pool scene.",
        revision_context,
        4,
    )

    assert stored.status == "revising"
    assert stored.quality_revision_count == 2
    assert stored.result["image_revision_feedback"] == (
        "Remove unrelated faces and keep one pool scene."
    )
    assert stored.result["image_revision_context"] == revision_context
    assert row.result == stored.result
    assert session.added[0].payload_json["status"] == "revising"


@pytest.mark.asyncio
async def test_image_generation_checkpoint_persists_and_clears_feedback(
    monkeypatch,
):
    import app.orchestrator as orchestrator_module

    record = ImageGenerationRecord(
        artifact_path="artifacts/revised.png",
        prompt="three women playing in one pool scene",
        negative_prompt="collage",
        width=768,
        height=1024,
        steps=30,
    )
    stored_result = {
        "status": "succeeded",
        "summary": "Previous subtask result.",
        "findings": [],
        "sources": [],
        "tool_runs": [],
        "questions": [],
        "error": "",
        "image_revision_feedback": "Replace the face grid.",
    }
    stored = SimpleNamespace(result=stored_result)

    class FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _model, _identity, **_kwargs):
            return stored

        async def scalar(self, _statement):
            return 0

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    async def assert_lease(*_args):
        return None

    row = SimpleNamespace(
        id=uuid.uuid4(),
        task_id=uuid.uuid4(),
        plan_version=1,
        result=stored_result,
    )
    review_id = uuid.uuid4()
    revision_context = {
        "trigger_quality_review_id": str(review_id),
        "source_artifact_paths": ["artifacts/previous.png"],
        "unmet_items": [
            {
                "item_id": "scene",
                "description": "Show one pool scene.",
                "evidence": "Many separate faces are visible.",
                "deficiency": "The composition is a face grid.",
            }
        ],
    }
    session = FakeSession()
    monkeypatch.setattr(
        orchestrator_module, "SessionFactory", lambda: session
    )
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )

    await orchestrator_module._checkpoint_image_generations(
        uuid.uuid4(),
        "worker-test",
        row,
        [record],
        record,
        revision_context,
        "test-image-model",
        4,
    )

    assert stored.result["image_generations"] == [
        record.model_dump(mode="json")
    ]
    assert stored.result["image_revision_feedback"] == ""
    assert stored.result["image_revision_completed"] is True
    assert row.result == stored.result
    assert orchestrator_module._stored_revision_feedback(stored.result) == ""
    history = next(
        item
        for item in session.added
        if isinstance(item, ImageGenerationRevision)
    )
    assert history.trigger_quality_review_id == review_id
    assert history.unmet_items == revision_context["unmet_items"]
    assert history.source_artifact_paths == ["artifacts/previous.png"]
    assert history.prompt == record.prompt
    progress_event = next(
        item for item in session.added if isinstance(item, Event)
    )
    assert progress_event.payload_json["activity"] == "image_generation"
    assert "prompt" not in progress_event.payload_json
    restored = orchestrator_module._restore_subtask_result(stored.result)
    assert restored.status == "succeeded"
    assert restored._image_generations == [record]


@pytest.mark.asyncio
async def test_final_image_review_revises_source_subtask(monkeypatch):
    import json

    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    subtask_id = uuid.uuid4()
    old_image = ImageGenerationRecord(
        artifact_path="artifacts/old-pool.png",
        prompt="three women at a pool",
        negative_prompt="collage",
        width=768,
        height=1024,
        steps=25,
    )
    new_image = ImageGenerationRecord(
        artifact_path="artifacts/new-pool.png",
        prompt="three women playing in one pool scene",
        negative_prompt="collage, multiple separate faces",
        width=768,
        height=1024,
        steps=30,
    )
    source_result = {
        "status": "succeeded",
        "summary": "Generated the requested image.",
        "findings": [],
        "sources": [],
        "tool_runs": [],
        "questions": [],
        "error": "",
        "image_generations": [old_image.model_dump(mode="json")],
    }
    row = SimpleNamespace(
        id=subtask_id,
        task_id=task_id,
        status="accepted",
        result=source_result,
        title="Generate the requested image",
        instruction="Draw three women playing in a pool.",
        artifact_goal="Show three women playing in one pool scene.",
        acceptance_checklist=[],
        dependencies=[],
        model_id="test-worker",
        quality_revision_count=0,
        tool_calls_used=1,
        tool_calls_limit=20,
        plan_version=1,
    )
    task = SimpleNamespace(
        config_snapshot={
            "max_concurrent_agents": 1,
            "max_agent_attempts": 1,
            "manager_model": "test-manager",
            "max_quality_revisions_per_task": 3,
            "max_quality_revisions_per_subtask": 2,
            "max_tool_calls_per_subtask": 20,
        }
    )
    conversation = [
        SimpleNamespace(
            sequence=1,
            role="user",
            content="Draw three women playing in a pool.",
        )
    ]

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, model, _identity):
            return task if model is Task else None

        async def scalars(self, _statement):
            return conversation

    plan = SimpleNamespace(
        summary="Generate the requested image.",
        final_checklist=[
            ChecklistItem(
                id="scene",
                description="Three women play together in one pool.",
            )
        ],
    )

    async def load_plan(*_args):
        return plan, 1, [row]

    async def quality_revision_count(*_args):
        return 0

    syntheses = []
    reviewed_paths = []
    revision_calls = []
    saved_verdicts = []
    saved_review_ids = []
    finished = {}
    reviews = [
        [
            {
                "item_id": "scene",
                "description": "Three women play together in one pool.",
                "required": True,
                "passed": False,
                "evidence": "Many separate faces are visible.",
                "deficiency": "The image is a grid of unrelated faces.",
                "revision_target": "image",
                "target_image_paths": [old_image.artifact_path],
            }
        ],
        [
            {
                "item_id": "scene",
                "description": "Three women play together in one pool.",
                "required": True,
                "passed": True,
                "evidence": "Three women are together in a pool scene.",
                "deficiency": "",
                "revision_target": "answer",
                "target_image_paths": [],
            }
        ],
    ]

    async def generate(**kwargs):
        syntheses.append(json.loads(kwargs["input_text"]))
        return IntegratedAnswer(
            answer="The revised image is ready.",
            caveats=[],
            artifacts=[],
        )

    async def review_checklist(**kwargs):
        reviewed_paths.append(kwargs["image_paths"])
        return reviews.pop(0)

    async def begin_image_revision(*_args, **_kwargs):
        row.quality_revision_count += 1

    async def execute_subtask(*_args, **kwargs):
        revision_calls.append(kwargs)
        revised = SubtaskResult(
            status="succeeded",
            summary="Generated a revised pool scene.",
            findings=[],
            sources=[],
            tool_runs=[],
            questions=[],
            error="",
        )
        revised._image_generations = [new_image]
        return revised

    async def review_subtask(_row, initial_result, *_args, **_kwargs):
        serialized = orchestrator_module._serialize_subtask_result(
            initial_result
        )
        _row.result = serialized
        _row.status = "accepted"
        return (
            serialized,
            "accepted",
            [],
        )

    async def save_quality_review(**kwargs):
        saved_verdicts.append(kwargs["verdict"])
        review_id = uuid.uuid4()
        saved_review_ids.append(review_id)
        return review_id

    async def finish_task(*_args, **kwargs):
        finished.update(kwargs)

    async def no_requeue(*_args):
        return False

    async def mark_subtasks_running(*_args):
        return None

    monkeypatch.setattr(
        orchestrator_module, "SessionFactory", FakeSession
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_load_or_create_plan",
        load_plan,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_requeue_if_new_messages",
        no_requeue,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_quality_revision_count",
        quality_revision_count,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_mark_subtasks_running",
        mark_subtasks_running,
    )
    monkeypatch.setattr(
        orchestrator_module, "_review_checklist", review_checklist
    )
    monkeypatch.setattr(
        orchestrator_module, "_save_quality_review", save_quality_review
    )
    monkeypatch.setattr(
        orchestrator_module, "_begin_image_revision", begin_image_revision
    )
    monkeypatch.setattr(
        orchestrator_module, "_execute_subtask", execute_subtask
    )
    monkeypatch.setattr(
        orchestrator_module, "_review_subtask_result", review_subtask
    )
    monkeypatch.setattr(
        orchestrator_module, "_finish_task", finish_task
    )
    monkeypatch.setattr(
        orchestrator_module, "_agent_options", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        orchestrator_module, "_format_conversation", lambda _messages: ""
    )
    monkeypatch.setattr(
        orchestrator_module.model_client, "generate", generate
    )

    await orchestrator_module.execute_task(task_id, job_id, "worker-test")

    assert len(revision_calls) == 1
    assert revision_calls[0]["previous_image_generations"] == [old_image]
    assert revision_calls[0]["revision_feedback"] == (
        "- Three women play together in one pool.: "
        "The image is a grid of unrelated faces."
    )
    assert revision_calls[0]["image_revision_context"] == {
        "trigger_quality_review_id": str(saved_review_ids[0]),
        "source_artifact_paths": [old_image.artifact_path],
        "unmet_items": [
            {
                "item_id": "scene",
                "description": "Three women play together in one pool.",
                "evidence": "Many separate faces are visible.",
                "deficiency": "The image is a grid of unrelated faces.",
            }
        ],
    }
    assert reviewed_paths == [
        [old_image.artifact_path],
        [new_image.artifact_path],
    ]
    assert syntheses[1]["generated_artifact_paths"] == [
        new_image.artifact_path
    ]
    assert saved_verdicts == ["revision_requested", "accepted"]
    assert finished["fully_completed"]
    assert finished["generated_artifact_paths"] == [
        new_image.artifact_path
    ]

    row.status = "revising"
    row.result["image_revision_feedback"] = ""
    row.result["image_revision_completed"] = True
    reviews.append(
        [
            {
                "item_id": "scene",
                "description": "Three women play together in one pool.",
                "required": True,
                "passed": True,
                "evidence": "Three women are together in a pool scene.",
                "deficiency": "",
                "revision_target": "answer",
                "target_image_paths": [],
            }
        ]
    )
    await orchestrator_module.execute_task(task_id, job_id, "worker-test")

    assert len(revision_calls) == 1
    assert reviewed_paths[-1] == [new_image.artifact_path]
    assert syntheses[-1]["generated_artifact_paths"] == [
        new_image.artifact_path
    ]


@pytest.mark.asyncio
async def test_tool_budget_limit_finalizes_partial_subtask(monkeypatch):
    import app.orchestrator as orchestrator_module

    row = SimpleNamespace(
        id=uuid.uuid4(),
        task_id=uuid.uuid4(),
        plan_version=1,
        tool_calls_used=19,
        tool_calls_limit=20,
        title="Research sources",
        instruction="Find source-backed facts",
        artifact_goal="Provide supported findings",
        acceptance_checklist=[],
        dependencies=[],
        model_id="test-worker",
    )
    calls = []

    async def assert_lease(*_args):
        return None

    async def record_activity(*_args, **_kwargs):
        return None

    async def generate(**options):
        calls.append(options)
        if len(calls) == 1:
            await options["on_tool_result"](
                "tavily_web_search",
                '{"query":"specific source"}',
                {"error": "tool_call_limit_exceeded"},
            )
            raise AgentExecutionError("agent_turn_limit_exceeded")
        assert options.get("tool_registry") is None
        assert options["max_tool_calls"] == 0
        return SubtaskResult(
            status="partial",
            summary="Partial evidence collected before the tool limit.",
            findings=["One supported finding."],
            sources=[],
            tool_runs=[],
            questions=[],
            error="tool_call_limit_exceeded",
        )

    monkeypatch.setattr(
        orchestrator_module, "_assert_lease", assert_lease
    )
    monkeypatch.setattr(
        orchestrator_module,
        "load_tavily_search_skill",
        lambda: "Skill-specific CLI search instructions.",
    )
    monkeypatch.setattr(
        orchestrator_module, "_record_subtask_activity", record_activity
    )
    monkeypatch.setattr(orchestrator_module.model_client, "generate", generate)

    result = await orchestrator_module._execute_subtask(
        row,
        SimpleNamespace(summary="Research plan"),
        {},
        [],
        uuid.uuid4(),
        "test-worker",
        2,
        config={"max_agent_turns": 24, "max_tool_calls_per_subtask": 20},
    )

    assert result.status == "partial"
    assert row.tool_calls_used == 20
    assert result.tool_runs[0].error_code == "tool_call_limit_exceeded"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_saved_subtask_result_keeps_image_checkpoint(monkeypatch):
    import app.orchestrator as orchestrator_module

    record = ImageGenerationRecord(
        artifact_path="artifacts/revised.png",
        prompt="revised image prompt",
        negative_prompt="collage",
        width=768,
        height=1024,
        steps=30,
    )
    stored = SimpleNamespace(
        status="revising",
        result={
            "image_generations": [record.model_dump(mode="json")],
            "image_revision_feedback": "Remove unrelated faces.",
        },
        error=None,
    )

    class FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _model, _identity):
            return stored

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    async def assert_lease(*_args):
        return None

    session = FakeSession()
    row = SimpleNamespace(id=uuid.uuid4(), title="Generate image")
    monkeypatch.setattr(
        orchestrator_module, "SessionFactory", lambda: session
    )
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )

    await orchestrator_module._save_subtask_result(
        uuid.uuid4(),
        uuid.uuid4(),
        "worker-test",
        row,
        {"status": "failed", "summary": "Model output failed."},
        None,
        3,
    )

    assert stored.status == "failed"
    assert stored.result["image_generations"] == [
        record.model_dump(mode="json")
    ]
    assert stored.result["image_revision_feedback"] == (
        "Remove unrelated faces."
    )
    assert all(
        "image_revision_feedback" not in event.payload_json
        for event in session.added
    )


@pytest.mark.asyncio
async def test_subtask_progress_events_are_tagged(monkeypatch):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    row = SimpleNamespace(id=uuid.uuid4(), title="Research")
    stored = SimpleNamespace(status="running", result=None, error=None)

    class FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _model, _identity):
            return stored

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    session = FakeSession()

    async def assert_lease(*_args):
        return None

    monkeypatch.setattr(orchestrator_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )

    await orchestrator_module._save_subtask_result(
        task_id,
        job_id,
        "worker-test",
        row,
        {"status": "succeeded", "summary": "Research complete"},
        None,
        4,
    )

    progress_events = [
        item for item in session.added if isinstance(item, Event)
    ]
    assert len(progress_events) == 2
    assert all(
        item.payload_json["conversation_sequence"] == 4
        for item in progress_events
    )


@pytest.mark.asyncio
async def test_tool_reservation_emits_named_progress_event(monkeypatch):
    import app.orchestrator as orchestrator_module

    task_id = uuid.uuid4()
    subtask_id = uuid.uuid4()
    job_id = uuid.uuid4()
    task = SimpleNamespace(
        id=task_id,
        status="running",
        config_snapshot={"max_tool_calls_per_subtask": 20},
    )
    job = SimpleNamespace()
    subtask = SimpleNamespace(
        tool_calls_limit=20,
        tool_calls_used=0,
        plan_version=3,
    )

    class FakeSession:
        def __init__(self):
            self.values = [task, job, subtask]
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return self.values.pop(0)

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    session = FakeSession()
    monkeypatch.setattr(
        orchestrator_module, "SessionFactory", lambda: session
    )

    await orchestrator_module._reserve_subtask_tool_call(
        task_id,
        subtask_id,
        job_id,
        "worker-test",
        "tavily_web_search",
        7,
    )

    event = session.added[0]
    assert event.type == "agent.progress"
    assert event.payload_json["activity"] == "tool"
    assert event.payload_json["phase"] == "started"
    assert event.payload_json["tool_name"] == "tavily_web_search"
    assert event.payload_json["conversation_sequence"] == 7
    assert subtask.tool_calls_used == 1


@pytest.mark.asyncio
async def test_retryable_worker_failure_clears_task_error(monkeypatch):
    import app.worker as worker_module

    task_id = uuid.uuid4()
    job_id = uuid.uuid4()
    task = SimpleNamespace(
        config_snapshot={"max_agent_attempts": 3},
        status="running",
        error={"code": "agent_attempt_failed"},
        updated_at=None,
    )
    job = SimpleNamespace(
        status="leased",
        lease_owner="worker-test",
        available_at=None,
        lease_expires_at=None,
    )

    class FakeSession:
        def __init__(self):
            self.added = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def scalar(self, _statement):
            return job

        async def get(self, _model, _identity):
            return task

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

    session = FakeSession()
    monkeypatch.setattr(worker_module, "worker_id", "worker-test")
    monkeypatch.setattr(
        worker_module, "SessionFactory", lambda: session
    )
    monkeypatch.setattr(
        worker_module,
        "get_settings",
        lambda: SimpleNamespace(max_agent_attempts=3),
    )

    await worker_module.mark_failure(
        job_id, task_id, 1, RuntimeError("transient timeout")
    )

    assert task.status == "queued"
    assert task.error is None
    assert job.status == "pending"
    assert session.added[0].payload_json == {
        "status": "queued",
        "retry_scheduled": True,
        "attempt": 1,
        "max_attempts": 3,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_status", ["needs_attention", "failed"])
async def test_user_reply_requeues_recoverable_task(initial_status):
    task_id = uuid.uuid4()
    user = User(
        id=uuid.uuid4(),
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )
    task = Task(
        id=task_id,
        owner_id=user.id,
        instruction="Research options",
        status=initial_status,
        result="stale result",
        error={"code": "needs_input"},
        completion_report={"unmet_items": []},
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, 2]
            self.added = []

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        def add(self, value):
            self.added.append(value)

        async def flush(self):
            return None

        async def commit(self):
            return None

        async def refresh(self, _value):
            return None

    session = FakeSession()
    await add_message(
        task_id,
        MessageCreate(content="Focus on first-time founders."),
        user,
        session,
    )

    assert task.status == "queued"
    assert task.result is None
    assert task.error is None
    assert task.completion_report is None
    assert any(
        isinstance(item, Job) and item.status == "pending"
        for item in session.added
    )
    queued_event = next(
        item
        for item in session.added
        if isinstance(item, Event) and item.type == "task.status_changed"
    )
    assert queued_event.payload_json["conversation_sequence"] == 3


@pytest.mark.asyncio
async def test_partial_task_reply_clears_report_and_requeues():
    task_id = uuid.uuid4()
    user = User(
        id=uuid.uuid4(),
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )
    task = Task(
        id=task_id,
        owner_id=user.id,
        instruction="Research options",
        status="partially_completed",
        result="Partial answer",
        completion_report={"unmet_items": [{"item_id": "sources"}]},
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, 2]
            self.added = []

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        def add(self, value):
            self.added.append(value)

        async def flush(self):
            return None

        async def commit(self):
            return None

        async def refresh(self, _value):
            return None

    session = FakeSession()
    await add_message(
        task_id,
        MessageCreate(content="Use official sources."),
        user,
        session,
    )

    assert task.status == "queued"
    assert task.result is None
    assert task.error is None
    assert task.completion_report is None
    assert any(
        isinstance(item, Job) and item.status == "pending"
        for item in session.added
    )


@pytest.mark.asyncio
async def test_editing_user_message_rewinds_conversation_and_requeues_task():
    task_id = uuid.uuid4()
    user = User(
        id=uuid.uuid4(),
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )
    task = Task(
        id=task_id,
        owner_id=user.id,
        instruction="Original request",
        title="Original title",
        title_generated=True,
        status="succeeded",
        result="stale result",
        error={"code": "old_error"},
        completion_report={"unmet_items": []},
    )
    message = ConversationMessage(
        id=uuid.uuid4(),
        task_id=task_id,
        sequence=1,
        role="user",
        content="Original request",
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, message, message.id, 5]
            self.added = []
            self.statements = []

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        async def execute(self, statement):
            self.statements.append(statement)

        def add(self, value):
            self.added.append(value)

        async def commit(self):
            return None

        async def refresh(self, _value):
            return None

    session = FakeSession()
    updated = await edit_message(
        task_id,
        message.id,
        MessageCreate(content="Revised request"),
        user,
        session,
    )

    assert updated is message
    assert message.content == "Revised request"
    assert message.sequence == 6
    assert task.instruction == "Revised request"
    assert task.title is None
    assert not task.title_generated
    assert task.status == "queued"
    assert task.result is None
    assert task.error is None
    assert task.completion_report is None
    assert len(session.statements) == 2
    assert "UPDATE task_files SET role=" in str(session.statements[1])
    assert session.statements[1].compile().params["role"] == "source"
    assert any(
        isinstance(item, Job) and item.status == "pending"
        for item in session.added
    )
    rewound_event = next(
        item
        for item in session.added
        if isinstance(item, Event) and item.type == "conversation.rewound"
    )
    assert rewound_event.payload_json["previous_sequence"] == 1
    assert rewound_event.payload_json["conversation_sequence"] == 6


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_status",
    ["queued", "planning", "running", "waiting_approval", "cancelled"],
)
async def test_message_edit_rejects_active_or_cancelled_task(initial_status):
    task = Task(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        instruction="Research options",
        status=initial_status,
    )
    user = User(
        id=task.owner_id,
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )

    class FakeSession:
        async def scalar(self, _statement):
            return task

    with pytest.raises(HTTPException) as error:
        await edit_message(
            task.id,
            uuid.uuid4(),
            MessageCreate(content="Revised request"),
            user,
            FakeSession(),
        )

    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_assistant_message_cannot_be_edited():
    task_id = uuid.uuid4()
    user = User(
        id=uuid.uuid4(),
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )
    task = Task(
        id=task_id,
        owner_id=user.id,
        instruction="Research options",
        status="succeeded",
    )
    message = ConversationMessage(
        id=uuid.uuid4(),
        task_id=task_id,
        sequence=2,
        role="assistant",
        content="Previous answer",
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, message]

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

    with pytest.raises(HTTPException) as error:
        await edit_message(
            task_id,
            message.id,
            MessageCreate(content="Revised answer"),
            user,
            FakeSession(),
        )

    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_failed_task_file_upload_requeues_task(monkeypatch, tmp_path):
    task_id = uuid.uuid4()
    user = User(
        id=uuid.uuid4(),
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )
    task = Task(
        id=task_id,
        owner_id=user.id,
        instruction="Research options",
        status="failed",
        result="stale result",
        error={"code": "provider_unavailable"},
    )

    class FakeSession:
        def __init__(self):
            self.scalar_values = [task, None, 0]
            self.added = []

        async def scalar(self, _statement):
            return self.scalar_values.pop(0)

        def add(self, value):
            self.added.append(value)

        async def flush(self):
            return None

        async def commit(self):
            return None

        async def refresh(self, _value):
            return None

    async def write_workspace_stream(_task_id, _path, _stream):
        return 3

    monkeypatch.setattr(
        "app.routers.workspace.resolve_workspace_file",
        lambda *_args, **_kwargs: tmp_path / "brief.txt",
    )
    monkeypatch.setattr(
        "app.routers.workspace.write_workspace_stream", write_workspace_stream
    )
    request = Request(
        {"type": "http", "method": "POST", "path": "/", "headers": []}
    )
    session = FakeSession()

    await upload_task_file(task_id, request, "brief.txt", user, session)

    assert task.status == "queued"
    assert task.result is None
    assert task.error is None
    assert any(
        isinstance(item, Job) and item.status == "pending"
        for item in session.added
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_status", ["cancelled", "waiting_approval"])
async def test_terminal_or_approval_task_rejects_follow_up_message(
    initial_status,
):
    task = Task(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        instruction="Research options",
        status=initial_status,
        pending_tool_approval=(
            {"request_id": str(uuid.uuid4())}
            if initial_status == "waiting_approval"
            else None
        ),
    )
    user = User(
        id=task.owner_id,
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )

    class FakeSession:
        async def scalar(self, _statement):
            return task

    with pytest.raises(HTTPException) as error:
        await add_message(
            task.id,
            MessageCreate(content="Continue this task."),
            user,
            FakeSession(),
        )

    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_tool_budget_approval_cannot_be_bypassed_by_file_upload():
    task = Task(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        instruction="Research options",
        status="waiting_approval",
        pending_tool_approval={"request_id": str(uuid.uuid4())},
    )
    user = User(
        id=task.owner_id,
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )

    class FakeSession:
        async def scalar(self, _statement):
            return task

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {"type": "http", "method": "POST", "path": "/", "headers": []},
        receive,
    )
    with pytest.raises(HTTPException) as error:
        await upload_task_file(
            task.id, request, "extra.md", user, FakeSession()
        )

    assert error.value.status_code == 409


def test_passwords_are_hashed_and_verifiable():
    value = "a-long-example-password"
    hashed = hash_password(value)
    assert hashed != value
    assert verify_password(value, hashed)
    assert not verify_password("another-password", hashed)


@pytest.mark.asyncio
async def test_request_validation_does_not_echo_submitted_secrets(monkeypatch):
    monkeypatch.setenv("MANAGER_MODEL", "test-manager")
    monkeypatch.setenv("WORKER_MODEL", "test-worker")
    from app.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/auth/login",
            json={"email": "invalid", "password": "do-not-echo-this"},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed"}
    assert "do-not-echo-this" not in response.text


class FakeAccessKeySession:
    def __init__(self, api_key=None, user=None):
        self.api_key = api_key
        self.user = user

    async def scalar(self, statement):
        return self.api_key

    async def get(self, model, user_id):
        return self.user


def bearer_request(token: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/tasks",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
            "query_string": b"",
        }
    )


@pytest.mark.asyncio
async def test_bearer_access_key_authenticates_its_owner():
    token = "issued-once-token"
    user_id = uuid.uuid4()
    user = User(
        id=user_id,
        email="person@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        must_change_password=False,
    )
    api_key = ApiAccessKey(
        id=uuid.uuid4(),
        user_id=user_id,
        key_hash=token_digest(token),
        revoked_at=None,
    )
    session = FakeAccessKeySession(api_key=api_key, user=user)

    authenticated = await get_task_user(bearer_request(token), db=session)

    assert authenticated is user


@pytest.mark.asyncio
async def test_invalid_bearer_key_is_rejected():
    session = FakeAccessKeySession()

    with pytest.raises(HTTPException) as error:
        await get_task_user(bearer_request("invalid-token"), db=session)

    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_bearer_write_does_not_require_browser_csrf_cookie():
    request = bearer_request("issued-once-token")

    await require_task_csrf(request, db=FakeAccessKeySession())


class SmallOutput(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_model_json_fallback_when_schema_mode_is_unsupported(
    sdk_transport,
):
    transport = sdk_transport(
        [
            httpx2.Response(
                400, json={"error": {"message": "json_schema unsupported"}}
            ),
            '{"answer":"validated"}',
        ]
    )

    result = await StructuredModelClient().generate(
        model="test-model",
        schema_name="small_output",
        schema=SmallOutput,
        system_prompt="Return a short answer.",
        input_text="Answer the prompt.",
        max_attempts=2,
    )

    assert result.answer == "validated"
    assert str(transport.clients[0].base_url) == "https://api.openai.com/v1/"
    requests = transport.requests
    assert len(requests) == 2
    assert "text" in requests[0]
    assert not requests[1].get("text", {}).get("format")
    assert all(client.is_closed() for client in transport.clients)


@pytest.mark.asyncio
async def test_tool_call_is_validated_executed_and_returned_to_model(
    sdk_transport,
):
    calls = []

    class ToolArguments(BaseModel):
        query: str

    async def search(arguments):
        calls.append(arguments.query)
        return {
            "results": [{"title": "Example", "url": "https://example.org"}]
        }

    registry = ToolRegistry(
        [
            ToolRegistration(
                name="test_search",
                version="1",
                description="Search test data",
                risk_class="read_only",
                timeout_seconds=5,
                arguments_model=ToolArguments,
                handler=search,
            )
        ]
    )
    transport = sdk_transport(
        [
            [
                {
                    "id": "function-1",
                    "type": "function_call",
                    "name": "test_search",
                    "arguments": '{"query":"latest release"}',
                    "call_id": "call-1",
                }
            ],
            '{"answer":"Release notes found."}',
        ]
    )
    lease_checks = []
    captured_results = []

    async def check_lease(_tool_name):
        lease_checks.append(True)

    result = await StructuredModelClient().generate(
        model="test-model",
        schema_name="small_output",
        schema=SmallOutput,
        system_prompt="Use tools if needed.",
        input_text="Find recent release information.",
        tool_registry=registry,
        before_tool_call=check_lease,
        on_tool_result=lambda name, arguments, value: captured_results.append(
            (name, arguments, value)
        ),
    )

    assert result.answer == "Release notes found."
    assert calls == ["latest release"]
    assert len(lease_checks) == 1
    assert captured_results[0][0] == "test_search"
    requests = transport.requests
    assert requests[0]["tools"][0]["name"] == "test_search"
    assert not requests[1].get("previous_response_id")
    assert any(
        item.get("call_id") == "call-1"
        and item["type"] == "function_call_output"
        for item in requests[1]["input"]
    )
    assert transport.credentials == ["Bearer test-key-1", "Bearer test-key-2"]
    assert all(request["store"] is False for request in requests)
    assert all(client.max_retries == 0 for client in transport.clients)
    assert all(client.is_closed() for client in transport.clients)


@pytest.mark.asyncio
async def test_tool_registry_rejects_unknown_and_invalid_arguments():
    invoked = []

    async def handler(_arguments):
        invoked.append(True)
        return {"ok": True}

    registry = ToolRegistry([TavilySearchTool().registration()])

    invalid = await registry.execute(
        "tavily_web_search", '{"query":"", "max_results":99}'
    )
    sensitive = await registry.execute(
        "tavily_web_search",
        '{"query":"contact admin@example.org", "max_results":2}',
    )
    unknown = await registry.execute("unknown", "{}")

    assert invalid == {"error": "invalid_tool_arguments"}
    assert sensitive == {"error": "invalid_tool_arguments"}
    assert unknown == {"error": "tool_not_allowed"}
    assert invoked == []


@pytest.mark.asyncio
async def test_tool_registry_blocks_non_read_only_tools():
    invoked = []

    class ToolArguments(BaseModel):
        value: str

    async def handler(_arguments):
        invoked.append(True)
        return {"ok": True}

    registry = ToolRegistry(
        [
            ToolRegistration(
                name="write_tool",
                version="1",
                description="A future write tool",
                risk_class="writes_data",
                timeout_seconds=5,
                arguments_model=ToolArguments,
                handler=handler,
            )
        ]
    )

    result = await registry.execute("write_tool", '{"value":"change"}')

    assert registry.openai_definitions() == []
    assert result == {"error": "tool_requires_approval"}
    assert invoked == []


@pytest.mark.asyncio
async def test_tavily_search_uses_worker_key_and_filters_cli_results(
    monkeypatch,
):
    import json

    import app.tools as tools_module

    payload = {
        "results": [
            {
                "title": "Useful result",
                "url": "https://example.org/article",
                "content": "x" * 10_500,
                "score": 0.9,
                "published_date": "2026-10-12",
            },
            {
                "title": "Invalid local result",
                "url": "file:///etc/passwd",
                "content": "Must be excluded.",
                "score": 0.2,
            },
        ]
    }
    request_values = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self, input=None):
            request_values["input"] = input
            return json.dumps(payload).encode(), None

        async def wait(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    async def create_subprocess_exec(*command, **kwargs):
        request_values["command"] = command
        request_values.update(kwargs)
        return FakeProcess()

    class NoopLimiter:
        async def acquire(self):
            return None

    monkeypatch.setattr(
        tools_module.asyncio,
        "create_subprocess_exec",
        create_subprocess_exec,
    )
    monkeypatch.setenv("TAVILY_API_KEY", "stale-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")
    monkeypatch.setattr(
        tools_module, "read_tavily_api_key", lambda: "worker-secret"
    )
    monkeypatch.setattr(
        tools_module,
        "get_settings",
        lambda: SimpleNamespace(
            tavily_timeout_seconds=20,
            max_agent_attempts=1,
        ),
    )
    tool = TavilySearchTool()
    tool._limiter = NoopLimiter()

    result = await tool._search(
        TavilySearchArguments(query="  current news  ", max_results=2)
    )

    assert request_values["command"] == (
        "tvly",
        "search",
        "-",
        "--max-results",
        "2",
        "--json",
    )
    assert request_values["input"] == b"current news"
    assert request_values["env"]["TAVILY_API_KEY"] == "worker-secret"
    assert "OPENAI_API_KEY" not in request_values["env"]
    assert request_values["stderr"] == tools_module.asyncio.subprocess.DEVNULL
    assert tool.max_response_bytes == 1_024_000
    assert len(result["results"]) == 1
    assert result["results"][0]["url"] == "https://example.org/article"
    assert len(result["results"][0]["content"]) == 10_000


def test_tavily_search_skill_loader_strips_frontmatter(tmp_path, monkeypatch):
    import app.tools as tools_module

    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        "---\nname: tavily-search\nallowed-tools: Bash(tvly *)\n---\n"
        "# Search\nUse the CLI for concise web searches.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        tools_module,
        "get_settings",
        lambda: SimpleNamespace(tavily_search_skill_path=str(skill_file)),
    )

    instructions = tools_module.load_tavily_search_skill()

    assert instructions == "# Search\nUse the CLI for concise web searches."
    assert "allowed-tools" not in instructions


def test_private_skill_loader_reads_live_updates(tmp_path, monkeypatch):
    import app.tools as tools_module

    skill_root = tmp_path / "private_skills"
    skill_directory = skill_root / "deep-research"
    skill_directory.mkdir(parents=True)
    skill_file = skill_directory / "SKILL.md"
    skill_file.write_text(
        "---\nname: deep-research\ndescription: 'Research carefully.'\n---\n"
        "# First version\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        tools_module,
        "get_settings",
        lambda: SimpleNamespace(private_skills_path=str(skill_root)),
    )

    assert tools_module.list_private_skills() == [
        {"name": "deep-research", "description": "Research carefully."}
    ]
    assert tools_module.load_private_skill("deep-research") == (
        "# First version"
    )

    skill_file.write_text(
        "---\nname: deep-research\ndescription: 'Research carefully.'\n---\n"
        "# Updated version\n",
        encoding="utf-8",
    )
    assert tools_module.load_private_skill("deep-research") == (
        "# Updated version"
    )


def test_optional_private_skill_falls_back_only_when_directory_is_missing(
    tmp_path, monkeypatch
):
    import app.tools as tools_module

    skill_root = tmp_path / "private_skills"
    skill_root.mkdir()
    monkeypatch.setattr(
        tools_module,
        "get_settings",
        lambda: SimpleNamespace(private_skills_path=str(skill_root)),
    )

    assert (
        tools_module.load_private_skill_if_present("image-generation")
        is None
    )

    skill_directory = skill_root / "image-generation"
    skill_directory.mkdir()
    (skill_directory / "SKILL.md").write_text(
        "This is not a valid skill.", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="private_skill_invalid"):
        tools_module.load_private_skill_if_present("image-generation")


@pytest.mark.asyncio
async def test_tavily_cli_timeout_stops_child_process(monkeypatch):
    import asyncio

    import app.tools as tools_module

    class FakeProcess:
        returncode = None
        killed = False

        async def communicate(self, input=None):
            raise asyncio.TimeoutError

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            return self.returncode

    process = FakeProcess()

    async def create_subprocess_exec(*_command, **_kwargs):
        return process

    monkeypatch.setattr(
        tools_module.asyncio,
        "create_subprocess_exec",
        create_subprocess_exec,
    )
    monkeypatch.setattr(
        tools_module, "read_tavily_api_key", lambda: "worker-secret"
    )
    monkeypatch.setattr(
        tools_module,
        "get_settings",
        lambda: SimpleNamespace(
            tavily_timeout_seconds=1,
            max_agent_attempts=1,
        ),
    )

    with pytest.raises(RuntimeError, match="^tavily_timeout$"):
        await tools_module.TavilySearchTool()._search(
            TavilySearchArguments(query="current news", max_results=1)
        )

    assert process.killed
    assert process.returncode == -9
