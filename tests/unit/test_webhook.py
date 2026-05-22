import asyncio
import json
import pytest
import respx
import httpx
from nio_mcp.webhook import WebhookDispatcher
from nio_mcp.models import MessageRecord


RECORD = MessageRecord(
    event_id="$abc:example.org",
    room_id="!room:example.org",
    sender="@alice:example.org",
    sender_name="Alice",
    body="Hello",
    timestamp=1700000000000,
)


@pytest.fixture
def dispatcher():
    return WebhookDispatcher(queue_maxsize=3)


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
    # Fill queue to capacity
    for i in range(3):
        q.put_nowait(json.dumps({"body": f"old-{i}"}))
    # Dispatch one more — should evict oldest
    await dispatcher.dispatch(RECORD)
    items = []
    while not q.empty():
        items.append(json.loads(q.get_nowait()))
    # Queue should have 3 items (maxsize); last item is the new RECORD
    assert len(items) == 3
    assert items[-1]["event_id"] == RECORD.event_id


async def test_dispatch_no_http_post_when_no_url(dispatcher):
    q = dispatcher.subscribe()
    # No webhook_url set — should complete without error
    await dispatcher.dispatch(RECORD)
    assert not q.empty()


@respx.mock
async def test_dispatch_posts_to_webhook_url():
    route = respx.post("http://example.com/hook").mock(return_value=httpx.Response(200))
    d = WebhookDispatcher(webhook_url="http://example.com/hook", queue_maxsize=10)
    await d.dispatch(RECORD)
    assert route.called


@respx.mock
async def test_dispatch_includes_hmac_signature():
    route = respx.post("http://example.com/hook").mock(return_value=httpx.Response(200))
    d = WebhookDispatcher(
        webhook_url="http://example.com/hook",
        webhook_secret="mysecret",
        queue_maxsize=10,
    )
    await d.dispatch(RECORD)
    assert route.called
    request = route.calls.last.request
    assert "X-Nio-MCP-Signature" in request.headers
    assert request.headers["X-Nio-MCP-Signature"].startswith("sha256=")


@respx.mock
async def test_dispatch_webhook_failure_does_not_raise():
    respx.post("http://example.com/hook").mock(side_effect=httpx.ConnectError("refused"))
    d = WebhookDispatcher(webhook_url="http://example.com/hook", queue_maxsize=10)
    await d.dispatch(RECORD)  # must not propagate


@respx.mock
async def test_dispatch_http_error_status_does_not_raise():
    respx.post("http://example.com/hook").mock(return_value=httpx.Response(503))
    d = WebhookDispatcher(webhook_url="http://example.com/hook", queue_maxsize=10)
    await d.dispatch(RECORD)  # must not propagate


@respx.mock
async def test_dispatch_webhook_failure_still_delivers_to_subscribers():
    respx.post("http://example.com/hook").mock(side_effect=httpx.ConnectError("refused"))
    d = WebhookDispatcher(webhook_url="http://example.com/hook", queue_maxsize=10)
    q = d.subscribe()
    await d.dispatch(RECORD)
    assert not q.empty()
    data = json.loads(q.get_nowait())
    assert data["event_id"] == RECORD.event_id


@respx.mock
async def test_dispatch_webhook_failure_logs_warning(caplog):
    import logging
    respx.post("http://example.com/hook").mock(side_effect=httpx.ConnectError("refused"))
    d = WebhookDispatcher(webhook_url="http://example.com/hook", queue_maxsize=10)
    with caplog.at_level(logging.WARNING, logger="nio_mcp.webhook"):
        await d.dispatch(RECORD)
    assert any("webhook" in r.message.lower() or "http" in r.message.lower() for r in caplog.records)
