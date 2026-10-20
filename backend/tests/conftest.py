import os
import json
from types import SimpleNamespace

import httpx2
import pytest
from openai import AsyncOpenAI

os.environ.setdefault("MANAGER_MODEL", "test-manager")
os.environ.setdefault("WORKER_MODEL", "test-worker")


@pytest.fixture
def sdk_transport(monkeypatch):
    import app.openai_client as client_module

    def configure(outputs):
        pending = iter(outputs)
        requests = []
        credentials = []
        clients = []
        key_reads = []

        def read_key():
            key = f"test-key-{len(key_reads) + 1}"
            key_reads.append(key)
            return key

        def respond(request):
            requests.append(json.loads(request.content))
            credentials.append(request.headers["authorization"])
            output = next(pending)
            if isinstance(output, Exception):
                raise output
            if isinstance(output, httpx2.Response):
                return output
            if isinstance(output, str):
                output = [
                    {
                        "id": f"msg-{len(requests)}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": output,
                                "annotations": [],
                            }
                        ],
                    }
                ]
            return httpx2.Response(
                200,
                json={
                    "id": f"response-{len(requests)}",
                    "object": "response",
                    "created_at": 0,
                    "model": "test-model",
                    "status": "completed",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": output,
                },
            )

        def create_client(**kwargs):
            client = AsyncOpenAI(
                **kwargs,
                http_client=httpx2.AsyncClient(
                    transport=httpx2.MockTransport(respond)
                ),
            )
            clients.append(client)
            return client

        monkeypatch.setattr(client_module, "AsyncOpenAI", create_client)
        monkeypatch.setattr(client_module, "read_openai_api_key", read_key)
        monkeypatch.setattr(
            client_module,
            "get_settings",
            lambda: SimpleNamespace(
                openai_base_url="",
                max_agent_attempts=3,
                agent_timeout_seconds=30,
            ),
        )
        return SimpleNamespace(
            requests=requests,
            credentials=credentials,
            clients=clients,
            key_reads=key_reads,
        )

    return configure
