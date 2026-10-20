import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, TypeVar

from agents import (
    Agent,
    AgentsException,
    FunctionTool,
    MaxTurnsExceeded,
    Model,
    ModelBehaviorError,
    ModelSettings,
    RunConfig,
    Runner,
    UserError,
    set_tracing_disabled,
)
from agents.items import ModelResponse, TResponseStreamEvent
from agents.models.interface import ModelTracing
from agents.models.openai_responses import OpenAIResponsesModel

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.tools import ToolRegistry, ToolResultCallback
from app.vault import read_openai_api_key

ModelOutput = TypeVar("ModelOutput", bound=BaseModel)
DEFAULT_MAX_AGENT_TURNS = 24
DEFAULT_MAX_TOOL_CALLS = 20
SDK_VERSION = "0.24.0"
AGENT_DEFINITION_VERSION = "1"

set_tracing_disabled(True)
logging.getLogger("openai.agents").disabled = True


class AgentExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentExecutionContext:
    task_id: str | None = None
    plan_version: int | None = None
    subtask_id: str | None = None
    conversation_sequence: int | None = None
    sdk_version: str = SDK_VERSION
    definition_version: str = AGENT_DEFINITION_VERSION


class VaultResponsesModel(Model):
    def __init__(self, model, attempts, before_model_call):
        self.model = model
        self.attempts = attempts
        self.retries_remaining = attempts - 1
        self.before_model_call = before_model_call
        self.structured_output_supported = True

    def allow_retry(self) -> bool:
        if self.retries_remaining <= 0:
            return False
        self.retries_remaining -= 1
        return True

    async def get_response(self, **request) -> ModelResponse:
        settings = get_settings()
        for attempt in range(self.attempts):
            if self.before_model_call is not None:
                await self.before_model_call()
            try:
                api_key = await asyncio.to_thread(read_openai_api_key)
            except Exception:
                raise AgentExecutionError("vault_key_unavailable") from None
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=settings.openai_base_url
                or "https://api.openai.com/v1",
                max_retries=0,
                timeout=settings.agent_timeout_seconds,
            )
            try:
                prepared = dict(request)
                prepared["tracing"] = ModelTracing.DISABLED
                if not self.structured_output_supported:
                    output_schema = prepared.get("output_schema")
                    if output_schema is not None:
                        prepared["system_instructions"] = (
                            (prepared.get("system_instructions") or "")
                            + "\nReturn only JSON matching this schema: "
                            + json.dumps(output_schema.json_schema())
                        )
                    prepared["output_schema"] = None
                return await OpenAIResponsesModel(
                    model=self.model, openai_client=client
                ).get_response(**prepared)
            except (APIConnectionError, APIStatusError) as exc:
                unsupported = (
                    isinstance(exc, APIStatusError)
                    and exc.status_code == 400
                    and self.structured_output_supported
                    and any(
                        term in str(exc).lower()
                        for term in (
                            "json_schema",
                            "structured output",
                            "text.format",
                        )
                    )
                )
                if unsupported and self.allow_retry():
                    self.structured_output_supported = False
                    continue
                retryable = (
                    isinstance(exc, APIConnectionError)
                    or exc.status_code == 429
                    or exc.status_code >= 500
                )
                if not retryable or not self.allow_retry():
                    raise AgentExecutionError(
                        "openai_request_failed"
                    ) from None
                retry_after = (
                    exc.response.headers.get("retry-after")
                    if isinstance(exc, APIStatusError)
                    else None
                )
                try:
                    delay = (
                        min(float(retry_after), 30.0)
                        if retry_after
                        else 2**attempt
                    )
                except ValueError:
                    delay = 2**attempt
                await asyncio.sleep(max(delay, 0) + random.uniform(0, 0.25))
            finally:
                await client.close()
        raise AgentExecutionError("openai_attempts_exhausted")

    def stream_response(
        self, **_request
    ) -> AsyncIterator[TResponseStreamEvent]:
        raise AgentExecutionError("agent_streaming_not_supported")


class StructuredModelClient:
    async def generate(
        self,
        *,
        model: str,
        schema_name: str,
        schema: type[ModelOutput],
        system_prompt: str,
        input_text: str,
        max_attempts: int | None = None,
        tool_registry: ToolRegistry | None = None,
        before_tool_call: Callable[[], Awaitable[None]] | None = None,
        before_model_call: Callable[[], Awaitable[None]] | None = None,
        on_tool_result: ToolResultCallback | None = None,
        max_turns: int = DEFAULT_MAX_AGENT_TURNS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        execution_context: AgentExecutionContext | None = None,
    ) -> ModelOutput:
        settings = get_settings()
        attempts = min(max(max_attempts or settings.max_agent_attempts, 1), 3)
        if not 1 <= max_turns <= 64 or not 0 <= max_tool_calls <= 100:
            raise AgentExecutionError("invalid_agent_limits")
        tool_calls_used = 0
        tool_budget_lock = asyncio.Lock()
        tool_failure: BaseException | None = None
        tools = []

        async def check_execution():
            if tool_failure is not None:
                raise tool_failure
            if before_model_call is not None:
                await before_model_call()

        def make_tool(definition):
            async def invoke(_context, arguments):
                nonlocal tool_calls_used, tool_failure
                try:
                    async with tool_budget_lock:
                        if tool_failure is not None:
                            raise tool_failure
                        if tool_calls_used >= max_tool_calls:
                            raise AgentExecutionError(
                                "tool_call_limit_exceeded"
                            )
                        if before_tool_call is not None:
                            await before_tool_call()
                        tool_calls_used += 1
                    result = await tool_registry.execute(
                        definition["name"], arguments
                    )
                    if on_tool_result is not None:
                        on_tool_result(definition["name"], arguments, result)
                    return json.dumps(result, ensure_ascii=False)
                except BaseException as exc:
                    tool_failure = exc
                    raise

            return FunctionTool(
                name=definition["name"],
                description=definition["description"],
                params_json_schema=definition["parameters"],
                on_invoke_tool=invoke,
            )

        if tool_registry is not None:
            tools = [
                make_tool(item) for item in tool_registry.openai_definitions()
            ]
        provider = VaultResponsesModel(model, attempts, check_execution)
        for attempt in range(attempts):
            agent = Agent(
                name=schema_name,
                instructions=system_prompt,
                model=provider,
                tools=tools,
                output_type=schema,
                model_settings=ModelSettings(
                    parallel_tool_calls=False, store=False
                ),
            )
            try:
                result = await Runner.run(
                    agent,
                    input_text,
                    context=execution_context or AgentExecutionContext(),
                    max_turns=max_turns,
                    run_config=RunConfig(
                        tracing_disabled=True,
                        trace_include_sensitive_data=False,
                    ),
                )
                validated = schema.model_validate(result.final_output)
                await check_execution()
                return validated
            except UserError:
                if tool_failure is not None:
                    raise tool_failure from None
                raise AgentExecutionError(
                    "agent_configuration_error"
                ) from None
            except MaxTurnsExceeded:
                raise AgentExecutionError(
                    "agent_turn_limit_exceeded"
                ) from None
            except (ModelBehaviorError, ValidationError):
                if not provider.allow_retry():
                    raise AgentExecutionError("invalid_model_output") from None
                input_text += (
                    "\nThe previous response did not match the required "
                    "schema. "
                    "Return corrected JSON."
                )
            except AgentsException:
                raise AgentExecutionError("agent_execution_failed") from None
        raise AgentExecutionError("openai_attempts_exhausted")
