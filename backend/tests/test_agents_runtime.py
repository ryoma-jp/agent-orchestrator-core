import asyncio
import base64
import json

import httpx2 as httpx
import pytest
from agents import Runner
from pydantic import BaseModel

from app.openai_client import (
    AgentExecutionContext,
    AgentExecutionError,
    StructuredModelClient,
)
from app.tools import ToolRegistration, ToolRegistry


class Answer(BaseModel):
    answer: str


class SearchArguments(BaseModel):
    query: str


def tool_call(name="search", arguments='{"query":"facts"}', call_id="call-1"):
    return [
        {
            "id": f"function-{call_id}",
            "type": "function_call",
            "name": name,
            "arguments": arguments,
            "call_id": call_id,
        }
    ]


def registry(calls):
    async def search(arguments):
        calls.append(arguments.query)
        return {"results": []}

    return ToolRegistry(
        [
            ToolRegistration(
                name="search",
                version="1",
                description="Search public facts",
                risk_class="read_only",
                timeout_seconds=1,
                arguments_model=SearchArguments,
                handler=search,
            )
        ]
    )


async def generate(**options):
    return await StructuredModelClient().generate(
        model="test-model",
        schema_name="answer",
        schema=Answer,
        system_prompt="Return a factual answer.",
        input_text="Find facts.",
        **options,
    )


@pytest.mark.asyncio
async def test_invalid_output_is_corrected_with_bounded_retries(sdk_transport):
    transport = sdk_transport(['{"wrong":true}', '{"answer":"corrected"}'])
    assert (await generate(max_attempts=2)).answer == "corrected"
    assert len(transport.requests) == 2
    assert "corrected JSON" in str(transport.requests[1]["input"])


def test_model_input_defaults_image_detail_to_auto():
    from app.openai_client import _model_input

    model_input = _model_input("review", [b"image"])
    image_input = model_input[0]["content"][1]
    assert image_input["detail"] == "auto"


@pytest.mark.asyncio
async def test_openai_image_review_inputs_use_high_detail(sdk_transport):
    transport = sdk_transport(['{"answer":"reviewed"}'])
    await generate(
        provider="openai",
        input_images=[b"\x89PNG\r\n\x1a\nimage"],
    )
    image_input = transport.requests[0]["input"][0]["content"][1]
    assert image_input["type"] == "input_image"
    assert image_input["detail"] == "high"


@pytest.mark.asyncio
async def test_task_plan_accepts_catalog_agent_ids_through_sdk(sdk_transport):
    from app.agent_schemas import TaskPlan

    checklist = [{
        "id": "image",
        "description": "Check the image",
        "required": True,
        "verification_mode": "model",
    }]
    sdk_transport([json.dumps({
        "title": "Create image",
        "summary": "Generate and review an image",
        "final_checklist": checklist,
        "subtasks": [{
            "title": "Create image",
            "instruction": "Generate an image",
            "artifact_goal": "An image",
            "acceptance_checklist": checklist,
            "dependencies": [],
            "agent_id": "illustrator",
            "review_agent_id": "inspector",
            "skill_names": [],
        }],
    })])
    result = await StructuredModelClient().generate(
        model="test-model",
        provider="openai",
        schema_name="task_plan",
        schema=TaskPlan,
        system_prompt="Plan the task.",
        input_text="Create an image.",
        max_attempts=1,
    )

    assert result.subtasks[0].agent_id == "illustrator"
    assert result.subtasks[0].review_agent_id == "inspector"


@pytest.mark.asyncio
async def test_image_specialist_review_supports_indeterminate_sdk_output(
    sdk_transport,
):
    from app.agent_schemas import ImageSpecialistReview

    transport = sdk_transport(
        [
            '{"verdict":"indeterminate","evidence":"",'
            '"deficiency":"The detail is not visible.",'
            '"target_image_paths":[]}'
        ]
    )

    result = await StructuredModelClient().generate(
        model="test-model",
        provider="openai",
        schema_name="image_specialist_review",
        schema=ImageSpecialistReview,
        system_prompt="Review the supplied image.",
        input_text="Inspect the visible details.",
        input_images=[b"\x89PNG\r\n\x1a\nimage"],
        max_attempts=1,
    )

    assert result.verdict == "indeterminate"
    image_input = transport.requests[0]["input"][0]["content"][1]
    assert image_input["type"] == "input_image"
    assert image_input["detail"] == "high"


@pytest.mark.asyncio
async def test_invalid_output_stops_without_exposing_provider_output(
    sdk_transport,
):
    transport = sdk_transport(['{"secret":"not for logs"}'] * 3)
    with pytest.raises(AgentExecutionError, match="^invalid_model_output$"):
        await generate(max_attempts=3)
    assert len(transport.requests) == 3


@pytest.mark.asyncio
async def test_api_and_format_retries_share_one_budget(
    sdk_transport, monkeypatch
):
    async def no_wait(_delay):
        return None

    monkeypatch.setattr("app.openai_client.asyncio.sleep", no_wait)
    transport = sdk_transport(
        [
            '{"wrong":true}',
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            '{"wrong":true}',
        ]
    )
    with pytest.raises(AgentExecutionError, match="^invalid_model_output$"):
        await generate(max_attempts=3)
    assert len(transport.requests) == 3
    assert len(transport.key_reads) == 3


@pytest.mark.asyncio
async def test_read_timeout_retries_within_model_attempt_budget(
    sdk_transport, monkeypatch
):
    async def no_wait(_delay):
        return None

    monkeypatch.setattr("app.openai_client.asyncio.sleep", no_wait)
    transport = sdk_transport(
        [httpx.ReadTimeout("response timed out"), '{"answer":"recovered"}']
    )

    result = await generate(max_attempts=2)

    assert result.answer == "recovered"
    assert len(transport.requests) == 2
    assert len(transport.key_reads) == 2
    assert all(client.is_closed() for client in transport.clients)


@pytest.mark.asyncio
async def test_read_timeout_exhaustion_is_classified_and_bounded(
    sdk_transport, monkeypatch
):
    async def no_wait(_delay):
        return None

    monkeypatch.setattr("app.openai_client.asyncio.sleep", no_wait)
    transport = sdk_transport(
        [httpx.ReadTimeout("response timed out")] * 2
    )

    with pytest.raises(
        AgentExecutionError, match="^openai_request_timed_out$"
    ):
        await generate(max_attempts=2)

    assert len(transport.requests) == 2
    assert all(client.is_closed() for client in transport.clients)


@pytest.mark.asyncio
async def test_provider_error_is_sanitized_and_client_closed(sdk_transport):
    transport = sdk_transport(
        [
            httpx.Response(
                401, json={"error": {"message": "secret-provider-body"}}
            )
        ]
    )
    with pytest.raises(AgentExecutionError, match="^openai_request_failed$"):
        await generate()
    assert len(transport.requests) == 1
    assert all(client.is_closed() for client in transport.clients)


@pytest.mark.asyncio
async def test_vault_failure_does_not_fall_back_to_environment(
    sdk_transport, monkeypatch
):
    transport = sdk_transport([])

    def unavailable():
        raise RuntimeError("secret-in-vault-error")

    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    monkeypatch.setattr("app.openai_client.read_openai_api_key", unavailable)
    with pytest.raises(AgentExecutionError, match="^vault_key_unavailable$"):
        await generate()
    assert transport.requests == []


@pytest.mark.asyncio
async def test_lease_loss_prevents_model_request(sdk_transport):
    transport = sdk_transport([])

    async def lost():
        raise RuntimeError("job_lease_lost")

    with pytest.raises(RuntimeError, match="^job_lease_lost$"):
        await generate(before_model_call=lost)
    assert transport.key_reads == []


@pytest.mark.asyncio
async def test_lease_loss_at_tool_boundary_stops_runner(sdk_transport):
    transport = sdk_transport([tool_call()])
    calls = []

    async def lost(_tool_name):
        raise RuntimeError("job_lease_lost")

    with pytest.raises(RuntimeError, match="^job_lease_lost$"):
        await generate(tool_registry=registry(calls), before_tool_call=lost)
    assert calls == []
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_from_tool_boundary(sdk_transport):
    sdk_transport([tool_call()])
    calls = []

    async def cancelled(_tool_name):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await generate(
            tool_registry=registry(calls), before_tool_call=cancelled
        )
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("limit_source", ["runner", "database"])
async def test_tool_budget_exhaustion_returns_control_result(
    sdk_transport, limit_source
):
    transport = sdk_transport(
        [
            tool_call(call_id="call-1"),
            tool_call(call_id="call-2"),
            '{"answer":"partial result from existing evidence"}',
        ]
    )
    calls = []
    observations = []
    reservations = 0

    async def reserve(_tool_name):
        nonlocal reservations
        reservations += 1
        if reservations > 1:
            raise AgentExecutionError("tool_call_limit_exceeded")

    result = await generate(
        tool_registry=registry(calls),
        before_tool_call=reserve if limit_source == "database" else None,
        on_tool_result=lambda _name, _arguments, value: observations.append(
            value
        ),
        max_tool_calls=2 if limit_source == "database" else 1,
    )

    assert result.answer == "partial result from existing evidence"
    assert calls == ["facts"]
    assert len(observations) == 2
    assert observations[1]["error"] == "tool_call_limit_exceeded"
    assert len(transport.requests) == 3


@pytest.mark.asyncio
async def test_runner_turn_limit_does_not_become_success(sdk_transport):
    transport = sdk_transport([tool_call()])
    with pytest.raises(
        AgentExecutionError, match="^agent_turn_limit_exceeded$"
    ):
        await generate(tool_registry=registry([]), max_turns=1)
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_invalid_and_unknown_tools_cannot_execute(sdk_transport):
    transport = sdk_transport(
        [
            tool_call(arguments='{"wrong":"value"}'),
            '{"answer":"no results"}',
        ]
    )
    calls = []
    await generate(tool_registry=registry(calls))
    assert calls == []
    assert "invalid_tool_arguments" in str(transport.requests[1]["input"])


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected(sdk_transport):
    transport = sdk_transport([tool_call(name="not_registered")])
    calls = []
    with pytest.raises(AgentExecutionError, match="^invalid_model_output$"):
        await generate(tool_registry=registry(calls), max_attempts=1)
    assert calls == []
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_execution_context_is_local_and_tracing_disabled(
    sdk_transport, monkeypatch
):
    transport = sdk_transport(['{"answer":"first"}', '{"answer":"second"}'])
    runs = []
    run = Runner.run

    async def observe(agent, model_input, **options):
        runs.append(options)
        return await run(agent, model_input, **options)

    monkeypatch.setattr(Runner, "run", observe)
    first, second = await asyncio.gather(
        generate(execution_context=AgentExecutionContext(task_id="first")),
        generate(execution_context=AgentExecutionContext(task_id="second")),
    )
    assert {first.answer, second.answer} == {"first", "second"}
    assert {item["context"].task_id for item in runs} == {"first", "second"}
    assert all(item["run_config"].tracing_disabled for item in runs)
    assert all(
        not item["run_config"].trace_include_sensitive_data for item in runs
    )
    assert all(not item.get("session") for item in runs)
    assert len(transport.requests) == 2


@pytest.mark.asyncio
async def test_image_input_is_preserved_during_schema_retry(sdk_transport):
    image_bytes = b"\x89PNG\r\n\x1a\nreview image"
    transport = sdk_transport(['{"wrong":true}', '{"answer":"checked"}'])

    result = await generate(input_images=[image_bytes], max_attempts=2)

    expected_image_url = "data:image/png;base64," + base64.b64encode(
        image_bytes
    ).decode("ascii")
    for request in transport.requests:
        content = request["input"][0]["content"]
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"] == expected_image_url
    assert "corrected JSON" in transport.requests[1]["input"][0][
        "content"
    ][0]["text"]
    assert result.answer == "checked"


@pytest.mark.asyncio
async def test_inflight_tool_is_cancelled_with_runner(sdk_transport):
    transport = sdk_transport([tool_call()])
    started = asyncio.Event()
    finished = asyncio.Event()

    async def slow_search(_arguments):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    tools = ToolRegistry(
        [
            ToolRegistration(
                name="search",
                version="1",
                description="Search public facts",
                risk_class="read_only",
                timeout_seconds=30,
                arguments_model=SearchArguments,
                handler=slow_search,
            )
        ]
    )
    execution = asyncio.create_task(generate(tool_registry=tools))
    await asyncio.wait_for(started.wait(), timeout=5)
    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(execution, timeout=5)
    assert finished.is_set()
    assert len(transport.requests) == 1
