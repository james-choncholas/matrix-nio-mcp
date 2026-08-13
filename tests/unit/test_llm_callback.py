import json

import httpx
import pytest

from nio_mcp.llm_callback import (
    OpenWebUIChatClient,
    _assistant_message,
    _openwebui_api_root,
    create_llm_callback_client,
)


def _body(request: httpx.Request) -> dict:
    return json.loads(request.content)


@pytest.mark.parametrize(
    ("assistant_error", "expected_error"),
    [
        ({"content": "provider unavailable"}, "provider unavailable"),
        ({"message": "model failed"}, "model failed"),
        ("request failed", "request failed"),
    ],
)
def test_assistant_message_rejects_openwebui_error(assistant_error, expected_error):
    payload = {"chat": {"history": {"messages": {"assistant": {"error": assistant_error}}}}}
    with pytest.raises(RuntimeError, match=expected_error):
        _assistant_message(payload, "assistant")


async def test_openwebui_client_runs_agent_loop():
    requests: list[httpx.Request] = []
    assistant_message_id = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal assistant_message_id
        requests.append(request)
        if request.method == "POST" and request.url.path == "/api/v1/chats/new":
            payload = _body(request)
            assistant_message_id = payload["chat"]["history"]["currentId"]
            return httpx.Response(200, json={"id": "chat-1"})
        if request.method == "POST" and request.url.path == "/api/chat/completions":
            return httpx.Response(200, json=None)
        if request.method == "GET" and request.url.path == "/api/v1/chats/chat-1":
            return httpx.Response(
                200,
                json={
                    "chat": {
                        "history": {
                            "messages": {
                                assistant_message_id: {
                                    "content": "no relevant messages received, passing for now",
                                    "output": [
                                        {
                                            "type": "tool_call",
                                            "name": "search_messages",
                                        }
                                    ],
                                    "done": True,
                                }
                            }
                        }
                    }
                },
            )
        return httpx.Response(404)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenWebUIChatClient(
        base_url="https://ai.example.com/api/v1",
        bearer_token="secret-token",
        http_client=http,
    )

    result = await client.run(
        model="steven",
        prompt="New Matrix messages:\nAlice: Hello",
        request_options={"tool_ids": ["server:mcp:matrix"]},
    )

    assert result["content"] == "no relevant messages received, passing for now"
    assert result["output"][0]["name"] == "search_messages"
    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/api/v1/chats/new"),
        ("POST", "/api/chat/completions"),
        ("GET", "/api/v1/chats/chat-1"),
    ]

    completion = _body(requests[1])
    assert completion["model"] == "steven"
    assert completion["stream"] is True
    assert completion["chat_id"] == "chat-1"
    assert completion["id"] == assistant_message_id
    assert completion["tool_ids"] == ["server:mcp:matrix"]
    assert completion["messages"] == [
        {"role": "user", "content": "New Matrix messages:\nAlice: Hello"}
    ]
    assert completion["background_tasks"] == {
        "title_generation": False,
        "tags_generation": False,
        "follow_up_generation": False,
    }
    assert requests[0].headers["Authorization"] == "Bearer secret-token"
    await http.aclose()


async def test_openwebui_client_polls_async_session_before_reading(monkeypatch):
    task_checks = 0
    assistant_message_id = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal task_checks, assistant_message_id
        if request.url.path == "/api/v1/chats/new":
            assistant_message_id = _body(request)["chat"]["history"]["currentId"]
            return httpx.Response(200, json={"id": "chat-async"})
        if request.url.path == "/api/chat/completions":
            return httpx.Response(200, json={"status": True, "task_ids": ["task-1"]})
        if request.url.path == "/api/tasks/chat/chat-async":
            task_checks += 1
            return httpx.Response(
                200,
                json={"task_ids": ["task-1"] if task_checks == 1 else []},
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "chat": {
                        "history": {
                            "messages": {assistant_message_id: {"content": "done"}}
                        }
                    }
                },
            )
        return httpx.Response(200)

    monkeypatch.setattr("nio_mcp.llm_callback._POLL_INTERVAL_SECONDS", 0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenWebUIChatClient(base_url="https://ai.example.com", http_client=http)
    result = await client.run(
        model="steven",
        prompt="hello",
        request_options={"session_id": "api-session"},
    )
    assert result["content"] == "done"
    assert task_checks == 2
    await http.aclose()


async def test_openwebui_client_leaves_chat_when_completion_fails():
    deleted = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal deleted
        if request.url.path == "/api/v1/chats/new":
            return httpx.Response(200, json={"id": "chat-failed"})
        if request.url.path == "/api/chat/completions":
            return httpx.Response(503, text="upstream overloaded")
        if request.method == "DELETE":
            deleted = True
            return httpx.Response(200)
        return httpx.Response(404)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenWebUIChatClient(base_url="https://ai.example.com", http_client=http)
    with pytest.raises(httpx.HTTPStatusError):
        await client.run(model="steven", prompt="hello", request_options={})
    assert not deleted
    await http.aclose()


async def test_openwebui_client_rejects_protocol_field_overrides():
    client = OpenWebUIChatClient(base_url="https://ai.example.com")
    with pytest.raises(ValueError, match="stream"):
        await client.run(
            model="steven",
            prompt="hello",
            request_options={"stream": False},
        )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("https://ai.example.com", "https://ai.example.com/api"),
        ("https://ai.example.com/api", "https://ai.example.com/api"),
        ("https://ai.example.com/api/v1", "https://ai.example.com/api"),
        ("https://ai.example.com/api/v1/", "https://ai.example.com/api"),
    ],
)
def test_openwebui_api_root_normalizes_existing_config(configured, expected):
    assert _openwebui_api_root(configured) == expected


def test_factory_returns_none_when_callback_disabled():
    assert create_llm_callback_client(
        backend="openwebui_chat",
        base_url="",
        bearer_token="",
        timeout_seconds=300,
    ) is None


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported"):
        create_llm_callback_client(
            backend="responses",
            base_url="https://ai.example.com",
            bearer_token="",
            timeout_seconds=300,
        )
