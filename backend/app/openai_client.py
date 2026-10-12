import asyncio
import json
import random
from typing import Awaitable, Callable, TypeVar

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.tools import ToolRegistry
from app.vault import read_openai_api_key

T = TypeVar("T", bound=BaseModel)
MAX_TOOL_CALLS = 20


class StructuredModelClient:
    async def generate(
        self,
        *,
        model: str,
        schema_name: str,
        schema: type[T],
        system_prompt: str,
        input_text: str,
        max_attempts: int | None = None,
        tool_registry: ToolRegistry | None = None,
        before_tool_call: Callable[[], Awaitable[None]] | None = None,
        on_tool_result: Callable[
            [str, dict[str, object]], None
        ] | None = None,
    ) -> T:
        settings = get_settings()
        attempts = min(max(max_attempts or settings.max_agent_attempts, 1), 3)
        correction = ""
        structured_output_supported = True
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=True)
        for attempt in range(attempts):
            try:
                api_key = await asyncio.to_thread(read_openai_api_key)
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=(
                        settings.openai_base_url
                        or "https://api.openai.com/v1"
                    ),
                    max_retries=0,
                    timeout=settings.agent_timeout_seconds,
                )
                try:
                    system_content = system_prompt
                    if not structured_output_supported:
                        system_content = (
                            f"{system_prompt}\n"
                            "Return only JSON matching this schema: "
                            f"{schema_json}"
                        )
                    request = {
                        "model": model,
                        "input": [
                            {
                                "role": "system",
                                "content": system_content,
                            },
                            {
                                "role": "user",
                                "content": input_text + correction,
                            },
                        ],
                    }
                    if structured_output_supported:
                        request["text"] = {
                            "format": {
                                "type": "json_schema",
                                "name": schema_name,
                                "schema": schema.model_json_schema(),
                                "strict": True,
                            }
                        }
                    tool_definitions = (
                        tool_registry.openai_definitions()
                        if tool_registry is not None
                        else []
                    )
                    if tool_definitions:
                        request.update(
                            {
                                "tools": tool_definitions,
                                "tool_choice": "auto",
                                "parallel_tool_calls": False,
                            }
                        )
                    response = await client.responses.create(**request)
                    tool_calls_used = 0
                    while tool_definitions:
                        calls = [
                            item
                            for item in getattr(response, "output", [])
                            if getattr(item, "type", None) == "function_call"
                        ]
                        if not calls:
                            break
                        if tool_calls_used + len(calls) > MAX_TOOL_CALLS:
                            raise RuntimeError("tool_call_limit_exceeded")
                        tool_outputs = []
                        for call in calls:
                            if before_tool_call is not None:
                                await before_tool_call()
                            name = call.name
                            tool_result = await tool_registry.execute(
                                name, call.arguments
                            )
                            tool_calls_used += 1
                            if on_tool_result is not None:
                                on_tool_result(
                                    name, call.arguments, tool_result
                                )
                            tool_outputs.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": call.call_id,
                                    "output": json.dumps(
                                        tool_result, ensure_ascii=False
                                    ),
                                }
                            )
                        follow_up = {
                            "model": model,
                            "previous_response_id": response.id,
                            "input": tool_outputs,
                            "tools": tool_definitions,
                            "tool_choice": "auto",
                            "parallel_tool_calls": False,
                        }
                        if structured_output_supported:
                            follow_up["text"] = request["text"]
                        response = await client.responses.create(**follow_up)
                finally:
                    await client.close()
                return schema.model_validate_json(response.output_text)
            except ValidationError:
                if attempt + 1 == attempts:
                    raise
                correction = (
                    "\nThe previous response did not match the required "
                    "schema. Return corrected JSON."
                )
            except (APIConnectionError, APIStatusError) as exc:
                error_text = str(exc).lower()
                unsupported_schema = (
                    isinstance(exc, APIStatusError)
                    and structured_output_supported
                    and exc.status_code == 400
                    and any(
                        term in error_text
                        for term in (
                            "json_schema",
                            "structured output",
                            "text.format",
                        )
                    )
                )
                if unsupported_schema and attempt + 1 < attempts:
                    structured_output_supported = False
                    correction = (
                        "\nReturn valid JSON matching the required schema."
                    )
                    continue
                retryable = (
                    isinstance(exc, APIConnectionError)
                    or exc.status_code == 429
                    or exc.status_code >= 500
                )
                if not retryable or attempt + 1 == attempts:
                    raise RuntimeError("openai_request_failed") from exc
                retry_after = (
                    exc.response.headers.get("retry-after")
                    if isinstance(exc, APIStatusError)
                    else None
                )
                try:
                    delay = (
                        min(float(retry_after), 30.0)
                        if retry_after
                        else min(2**attempt, 20.0)
                    )
                except ValueError:
                    delay = min(2**attempt, 20.0)
                await asyncio.sleep(delay + random.uniform(0, 0.25))
        raise RuntimeError("openai_attempts_exhausted")
