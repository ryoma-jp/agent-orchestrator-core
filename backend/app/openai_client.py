import asyncio
import json
import random
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.vault import read_openai_api_key

T = TypeVar("T", bound=BaseModel)


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
                    response = await client.responses.create(**request)
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
