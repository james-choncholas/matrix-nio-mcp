import asyncio
import json
import pytest
import respx
import httpx

from nio_mcp.webhook import WebhookDispatcher, _render_per_msg, _render_prompt
from nio_mcp.models import MessageRecord


RECORD = MessageRecord(
    event_id="$abc:example.org",
    room_id="!room:example.org",
    room_name="Test Room",
    sender="@alice:example.org",
    sender_name="Alice",
    body="Hello",
    timestamp=1700000000000,
)

RECORD2 = MessageRecord(
    event_id="$def:example.org",
    room_id="!room:example.org",
    room_name="Test Room",
    sender="@bob:example.org",
    sender_name="Bob",
    body="World",
    timestamp=1700000001000,
)


@pytest.fixture
def dispatcher():
    return WebhookDispatcher(queue_maxsize=3)


# ── SSE subscriber mechanics ──────────────────────────────────────────────────

def test_subscribe_returns_bounded_queue(dispatcher):
    q = dispatcher.subscribe()
    assert q.maxsize == 3
    assert q in dispatcher._subscribers


def test_unsubscribe_removes_queue(dispatcher):
    q = dispatcher.subscribe()
    dispatcher.unsubscribe(q)
    assert q not in dispatcher._subscribers


def test_unsubscribe_unknown_queue_is_safe(dispatcher):
    q: asyncio.Queue = asyncio.Queue()
    dispatcher.unsubscribe(q)  # should not raise


async def test_dispatch_delivers_to_all_subscribers(dispatcher):
    q1 = dispatcher.subscribe()
    q2 = dispatcher.subscribe()
    await dispatcher.dispatch(RECORD)
    assert not q1.empty()
    assert not q2.empty()
    data1 = json.loads(q1.get_nowait())
    data2 = json.loads(q2.get_nowait())
    assert data1["event_id"] == RECORD.event_id
    assert data2["event_id"] == RECORD.event_id


async def test_dispatch_does_not_deliver_to_unsubscribed(dispatcher):
    q = dispatcher.subscribe()
    dispatcher.unsubscribe(q)
    await dispatcher.dispatch(RECORD)
    assert q.empty()


async def test_dispatch_full_queue_drops_oldest_not_newest(dispatcher):
    q = dispatcher.subscribe()
    for i in range(3):
        q.put_nowait(json.dumps({"body": f"old-{i}"}))
    await dispatcher.dispatch(RECORD)
    items = []
    while not q.empty():
        items.append(json.loads(q.get_nowait()))
    assert len(items) == 3
    assert items[-1]["event_id"] == RECORD.event_id


async def test_dispatch_no_llm_call_when_no_url(dispatcher):
    q = dispatcher.subscribe()
    await dispatcher.dispatch(RECORD)
    assert not q.empty()
    assert dispatcher._cooldown_task is None


# ── Prompt rendering ──────────────────────────────────────────────────────────

def test_render_per_msg_all_placeholders():
    result = _render_per_msg(
        "{sender_name} ({sender}) in {room_name} ({room}): {message}", RECORD
    )
    assert result == "Alice (@alice:example.org) in Test Room (!room:example.org): Hello"


def test_render_per_msg_subset_of_placeholders():
    result = _render_per_msg("{sender_name} said {message}", RECORD)
    assert result == "Alice said Hello"


def test_render_per_msg_body_with_braces_not_reinterpreted():
    record = MessageRecord(
        event_id="$x", room_id="!r", room_name="R",
        sender="@a", sender_name="A",
        body="use {sender} carefully",
        timestamp=0,
    )
    result = _render_per_msg("msg: {message}", record)
    assert result == "msg: use {sender} carefully"


def test_render_prompt_header_prepended_once():
    result = _render_prompt("Header:", "{message}", [RECORD, RECORD2])
    lines = result.splitlines()
    assert lines[0] == "Header:"
    assert lines[1] == "Hello"
    assert lines[2] == "World"


def test_render_prompt_no_header():
    result = _render_prompt("", "{message}", [RECORD, RECORD2])
    lines = result.splitlines()
    assert lines == ["Hello", "World"]


def test_render_prompt_single_message():
    result = _render_prompt("Hdr:", "{sender_name}: {message}", [RECORD])
    assert result == "Hdr:\nAlice: Hello"


def test_render_prompt_per_msg_applied_to_each_record():
    result = _render_prompt("", "{sender_name}", [RECORD, RECORD2])
    assert result == "Alice\nBob"


# ── LLM webhook call ──────────────────────────────────────────────────────────

@respx.mock
async def test_llm_called_with_chat_completions_format():
    route = respx.post("http://llm.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    d = WebhookDispatcher(
        webhook_url="http://llm.example.com/v1",
        bearer_token="test-token",
        prompt_header="Messages:",
        prompt_per_msg="{message}",
        model="gpt-4o-mini",
        cooldown_seconds=0.01,
        queue_maxsize=10,
    )
    await d.start()
    await d.dispatch(RECORD)
    await asyncio.sleep(0.05)
    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"][0]["role"] == "user"
    assert "Hello" in body["messages"][0]["content"]
    assert "Messages:" in body["messages"][0]["content"]


@respx.mock
async def test_llm_omits_authorization_header_when_no_token():
    route = respx.post("http://llm.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    d = WebhookDispatcher(
        webhook_url="http://llm.example.com/v1",
        bearer_token="",
        cooldown_seconds=0.01,
        queue_maxsize=10,
    )
    await d.start()
    await d.dispatch(RECORD)
    await asyncio.sleep(0.05)
    assert "authorization" not in route.calls.last.request.headers


@respx.mock
async def test_llm_includes_bearer_token():
    route = respx.post("http://llm.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    d = WebhookDispatcher(
        webhook_url="http://llm.example.com/v1",
        bearer_token="secret-token",
        cooldown_seconds=0.01,
        queue_maxsize=10,
    )
    await d.start()
    await d.dispatch(RECORD)
    await asyncio.sleep(0.05)
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer secret-token"


@respx.mock
async def test_llm_batches_multiple_messages_in_one_call():
    route = respx.post("http://llm.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    d = WebhookDispatcher(
        webhook_url="http://llm.example.com/v1",
        bearer_token="tok",
        prompt_header="",
        prompt_per_msg="{message}",
        cooldown_seconds=0.01,
        queue_maxsize=10,
    )
    await d.start()
    await d.dispatch(RECORD)
    await d.dispatch(RECORD2)
    await asyncio.sleep(0.05)
    assert route.call_count == 1
    body = json.loads(route.calls.last.request.content)
    content = body["messages"][0]["content"]
    assert "Hello" in content
    assert "World" in content


@respx.mock
async def test_cooldown_resets_on_new_message():
    route = respx.post("http://llm.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    d = WebhookDispatcher(
        webhook_url="http://llm.example.com/v1",
        bearer_token="tok",
        cooldown_seconds=0.05,
        queue_maxsize=10,
    )
    await d.start()
    await d.dispatch(RECORD)
    first_task = d._cooldown_task
    await asyncio.sleep(0.02)
    assert not route.called  # cooldown not yet expired
    await d.dispatch(RECORD2)  # resets the timer
    assert first_task is not d._cooldown_task  # new task created
    await asyncio.sleep(0.02)
    assert not route.called  # still within new cooldown window
    await asyncio.sleep(0.06)  # full cooldown after second message
    assert route.called


@respx.mock
async def test_llm_failure_does_not_raise():
    respx.post("http://llm.example.com/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("refused")
    )
    d = WebhookDispatcher(
        webhook_url="http://llm.example.com/v1",
        bearer_token="tok",
        cooldown_seconds=0.01,
        queue_maxsize=10,
    )
    await d.start()
    await d.dispatch(RECORD)
    await asyncio.sleep(0.05)  # must not propagate


@respx.mock
async def test_llm_http_error_status_does_not_raise():
    respx.post("http://llm.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(503)
    )
    d = WebhookDispatcher(
        webhook_url="http://llm.example.com/v1",
        bearer_token="tok",
        cooldown_seconds=0.01,
        queue_maxsize=10,
    )
    await d.start()
    await d.dispatch(RECORD)
    await asyncio.sleep(0.05)  # must not propagate


@respx.mock
async def test_llm_failure_still_delivers_to_sse_subscribers():
    respx.post("http://llm.example.com/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("refused")
    )
    d = WebhookDispatcher(
        webhook_url="http://llm.example.com/v1",
        bearer_token="tok",
        cooldown_seconds=0.01,
        queue_maxsize=10,
    )
    await d.start()
    q = d.subscribe()
    await d.dispatch(RECORD)
    assert not q.empty()
    data = json.loads(q.get_nowait())
    assert data["event_id"] == RECORD.event_id
    await asyncio.sleep(0.05)  # let the cooldown fire and fail; must not propagate


@respx.mock
async def test_llm_includes_tools_in_body_when_set():
    route = respx.post("http://llm.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    d = WebhookDispatcher(
        webhook_url="http://llm.example.com/v1",
        bearer_token="tok",
        cooldown_seconds=0.01,
        queue_maxsize=10,
        tools='{"tool_ids": ["server:mcp:myserver"]}',
    )
    await d.start()
    await d.dispatch(RECORD)
    await asyncio.sleep(0.05)
    body = json.loads(route.calls.last.request.content)
    assert body["tool_ids"] == ["server:mcp:myserver"]


@respx.mock
async def test_llm_omits_tools_from_body_when_not_set():
    route = respx.post("http://llm.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    d = WebhookDispatcher(
        webhook_url="http://llm.example.com/v1",
        bearer_token="tok",
        cooldown_seconds=0.01,
        queue_maxsize=10,
    )
    await d.start()
    await d.dispatch(RECORD)
    await asyncio.sleep(0.05)
    body = json.loads(route.calls.last.request.content)
    assert "tool_ids" not in body
    assert "tools" not in body


@respx.mock
async def test_llm_failure_logs_warning(caplog):
    import logging
    respx.post("http://llm.example.com/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("refused")
    )
    d = WebhookDispatcher(
        webhook_url="http://llm.example.com/v1",
        bearer_token="tok",
        cooldown_seconds=0.01,
        queue_maxsize=10,
    )
    await d.start()
    with caplog.at_level(logging.WARNING, logger="nio_mcp.webhook"):
        await d.dispatch(RECORD)
        await asyncio.sleep(0.05)
    assert any(
        "llm" in r.message.lower() or "webhook" in r.message.lower()
        for r in caplog.records
    )
