import uuid
from types import SimpleNamespace

import pytest
import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError
from starlette.requests import Request

from app.agent_schemas import PlannedSubtask, SubtaskResult, TaskPlan
from app.dependencies import get_task_user, require_task_csrf
from app.models import ApiAccessKey, ConversationMessage, Event, Job, Task, User
from app.openai_client import StructuredModelClient
from app.routers.tasks import add_message
from app.schemas import MessageCreate, TaskCreate
from app.security import hash_password, token_digest, verify_password


def test_task_instruction_is_bounded_and_nonempty():
    assert (
        TaskCreate(instruction="research this").instruction == "research this"
    )


def test_empty_task_instruction_is_rejected():
    try:
        TaskCreate(instruction="")
    except ValueError:
        return
    raise AssertionError("empty instruction was accepted")


def test_message_content_is_bounded_and_nonempty():
    assert MessageCreate(content="follow-up").content == "follow-up"


def test_task_plan_accepts_a_valid_dependency_graph():
    plan = TaskPlan(
        summary="Research and synthesize",
        completion_criteria=["Provide the answer"],
        subtasks=[
            PlannedSubtask(
                title="Research",
                instruction="Find facts",
                dependencies=[],
                skill_names=[],
            ),
            PlannedSubtask(
                title="Synthesis",
                instruction="Combine facts",
                dependencies=[0],
                skill_names=[],
            ),
        ],
    )
    assert plan.subtasks[1].dependencies == [0]


def test_task_plan_rejects_dependency_cycles():
    with pytest.raises(ValidationError, match="contains a cycle"):
        TaskPlan(
            summary="Invalid plan",
            completion_criteria=["Complete"],
            subtasks=[
                PlannedSubtask(
                    title="A",
                    instruction="A",
                    dependencies=[1],
                    skill_names=[],
                ),
                PlannedSubtask(
                    title="B",
                    instruction="B",
                    dependencies=[0],
                    skill_names=[],
                ),
            ],
        )


def test_subtask_result_requires_questions_for_user_input():
    result = SubtaskResult(
        status="needs_input",
        summary="More information is needed.",
        findings=[],
        questions=["Which audience should I focus on?"],
        error="",
    )
    assert result.questions == ["Which audience should I focus on?"]
    with pytest.raises(ValidationError, match="requires a question"):
        SubtaskResult(
            status="needs_input",
            summary="More information is needed.",
            findings=[],
            questions=[],
            error="",
        )


@pytest.mark.asyncio
async def test_user_questions_pause_task_and_are_saved_as_assistant_message(
    monkeypatch,
):
    import app.orchestrator as orchestrator_module

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

    monkeypatch.setattr(orchestrator_module, "SessionFactory", lambda: session)
    monkeypatch.setattr(
        orchestrator_module, "_assert_lease_in_session", assert_lease
    )

    await orchestrator_module._pause_for_user_input(
        task_id,
        job_id,
        lease_owner,
        1,
        [" Which audience should I focus on? ", "Which audience should I focus on?"],
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

    async def generate(**kwargs):
        captured.update(kwargs)
        return SubtaskResult(
            status="succeeded",
            summary="Completed",
            findings=[],
            questions=[],
            error="",
        )

    monkeypatch.setattr(orchestrator_module, "_assert_lease", assert_lease)
    monkeypatch.setattr(orchestrator_module.model_client, "generate", generate)

    await orchestrator_module._execute_subtask(
        SimpleNamespace(
            title="Clarify profile",
            instruction="Proceed",
            dependencies=[],
            model_id="test-worker",
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
    assert "do not ask again" in captured["system_prompt"]


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
async def test_user_reply_requeues_task_waiting_for_attention():
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
        status="needs_attention",
        result="stale result",
        error={"code": "needs_input"},
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
    monkeypatch,
):
    import app.openai_client as client_module

    clients = []

    class FakeResponses:
        def __init__(self):
            self.requests = []

        async def create(self, **request):
            self.requests.append(request)
            if "text" in request:
                response = httpx.Response(
                    400,
                    request=httpx.Request("POST", "https://example.invalid"),
                    json={"error": {"message": "json_schema unsupported"}},
                )
                from openai import APIStatusError

                raise APIStatusError(
                    "json_schema structured output unsupported",
                    response=response,
                    body=response.json(),
                )
            return SimpleNamespace(output_text='{"answer":"validated"}')

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.responses = FakeResponses()
            clients.append(self)

        async def close(self):
            return None

    monkeypatch.setattr(client_module, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(
        client_module,
        "get_settings",
        lambda: SimpleNamespace(
            openai_base_url="", max_agent_attempts=3,
            agent_timeout_seconds=30,
        ),
    )
    monkeypatch.setattr(
        client_module, "read_openai_api_key", lambda: "test-key"
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
    assert clients[0].kwargs["base_url"] == "https://api.openai.com/v1"
    requests = [
        request for client in clients for request in client.responses.requests
    ]
    assert len(requests) == 2
    assert "text" in requests[0]
    assert "text" not in requests[1]
