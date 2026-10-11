import uuid
from types import SimpleNamespace

import pytest
import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError
from starlette.requests import Request

from app.agent_schemas import PlannedSubtask, TaskPlan
from app.dependencies import get_task_user, require_task_csrf
from app.models import ApiAccessKey, User
from app.openai_client import StructuredModelClient
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
            self.responses = FakeResponses()
            clients.append(self)

        async def close(self):
            return None

    monkeypatch.setattr(client_module, "AsyncOpenAI", FakeClient)
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
    requests = [
        request for client in clients for request in client.responses.requests
    ]
    assert len(requests) == 2
    assert "text" in requests[0]
    assert "text" not in requests[1]
