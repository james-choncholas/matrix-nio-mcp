import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from collections import deque

from nio_mcp.matrix_client import MatrixMCPClient
from nio_mcp.models import MessageRecord
import nio


def _make_config(
    backfill_pages_max=2,
    backfill_limit=5,
    message_buffer_size=50,
):
    cfg = MagicMock()
    cfg.matrix_homeserver_url = "https://matrix.example.org"
    cfg.matrix_user_id = "@bot:example.org"
    cfg.matrix_device_id = "DEVID123"
    cfg.matrix_access_token = "syt_token"
    cfg.matrix_store_path = "/tmp/test_nio_store"
    cfg.backfill_pages_max = backfill_pages_max
    cfg.backfill_limit = backfill_limit
    cfg.message_buffer_size = message_buffer_size
    return cfg


def _make_room_messages_response(chunk, end=None):
    resp = MagicMock(spec=nio.RoomMessagesResponse)
    resp.chunk = chunk
    resp.end = end
    return resp


def _make_text_event(event_id="$evt:example.org", sender="@alice:example.org", body="Hello"):
    event = MagicMock(spec=nio.RoomMessageText)
    event.event_id = event_id
    event.sender = sender
    event.body = body
    event.server_timestamp = 1700000000000
    return event


def _make_initial_sync(room_id="!room:example.org", prev_batch="t1"):
    sync = MagicMock(spec=nio.SyncResponse)
    sync.next_batch = "next_batch_token"
    timeline = MagicMock()
    timeline.prev_batch = prev_batch
    room_info = MagicMock()
    room_info.timeline = timeline
    sync.rooms = MagicMock()
    sync.rooms.join = {room_id: room_info}
    return sync


@pytest.fixture
def mock_nio_client():
    with patch("nio_mcp.matrix_client.AsyncClient") as cls:
        instance = AsyncMock()
        cls.return_value = instance
        instance.restore_login = MagicMock()
        instance.add_event_callback = MagicMock()
        instance.rooms = {}
        yield instance


@pytest.fixture
def mock_makedirs():
    with patch("nio_mcp.matrix_client.os.makedirs") as m:
        yield m


@pytest.fixture
def vector_store():
    vs = AsyncMock()
    return vs


@pytest.fixture
def embedding_client():
    ec = AsyncMock()
    ec.embed.return_value = [0.1] * 1536
    ec.embed_batch.return_value = [[0.1] * 1536]
    return ec


@pytest.fixture
def webhook_dispatcher():
    wd = AsyncMock()
    return wd


@pytest.fixture
def client(mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher):
    cfg = _make_config()
    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    c._client = mock_nio_client
    return c


# --- start() behaviour ---

async def test_start_creates_store_dir_before_restore_login(
    mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher
):
    cfg = _make_config()
    call_order = []

    mock_makedirs.side_effect = lambda *a, **kw: call_order.append("makedirs")
    mock_nio_client.restore_login.side_effect = lambda **kw: call_order.append("restore_login")
    mock_nio_client.loaded_sync_token = None
    mock_nio_client.sync.return_value = _make_initial_sync()
    mock_nio_client.joined_rooms.return_value = MagicMock(rooms=[])

    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    with patch("nio_mcp.matrix_client.asyncio.create_task"):
        await c.start()

    assert call_order.index("makedirs") < call_order.index("restore_login")


async def test_start_calls_restore_login_with_env_credentials(
    mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher
):
    cfg = _make_config()
    mock_nio_client.loaded_sync_token = None
    mock_nio_client.sync.return_value = _make_initial_sync()
    mock_nio_client.joined_rooms.return_value = MagicMock(rooms=[])

    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    with patch("nio_mcp.matrix_client.asyncio.create_task"):
        await c.start()

    mock_nio_client.restore_login.assert_called_once_with(
        user_id="@bot:example.org",
        device_id="DEVID123",
        access_token="syt_token",
    )


async def test_start_passes_initial_sync_next_batch_to_sync_forever(
    mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher
):
    cfg = _make_config()
    initial_sync = _make_initial_sync()
    mock_nio_client.loaded_sync_token = None
    mock_nio_client.sync.return_value = initial_sync
    mock_nio_client.joined_rooms.return_value = MagicMock(rooms=[])

    sync_forever_kwargs = {}

    async def capturing_sync_forever(**kwargs):
        sync_forever_kwargs.update(kwargs)

    mock_nio_client.sync_forever = capturing_sync_forever

    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    with patch("nio_mcp.matrix_client.asyncio.create_task") as create_task:
        def capture_and_close(coro):
            coro.close()
            return MagicMock()

        create_task.side_effect = capture_and_close
        await c.start()
        assert create_task.called


async def test_start_resumes_from_stored_token_on_restart(
    mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher
):
    cfg = _make_config()
    mock_nio_client.loaded_sync_token = "stored_t99"

    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    with patch("nio_mcp.matrix_client.asyncio.create_task") as create_task:
        def capture_and_close(coro):
            coro.close()
            return MagicMock()

        create_task.side_effect = capture_and_close
        await c.start()

    # On restart: skip initial sync and backfill entirely
    mock_nio_client.sync.assert_not_called()
    mock_nio_client.joined_rooms.assert_not_called()
    # sync_forever must be started from the stored token
    assert create_task.called
    mock_nio_client.sync_forever.assert_called_once_with(since="stored_t99", timeout=30000)


# --- backfill pagination ---

async def test_backfill_room_stops_when_end_is_none(client, mock_nio_client):
    event = _make_text_event()
    resp1 = _make_room_messages_response([event], end="page2")
    resp2 = _make_room_messages_response([_make_text_event("$b:x", body="B")], end=None)
    mock_nio_client.room_messages = AsyncMock(side_effect=[resp1, resp2])
    client._embedding_client.embed_batch = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])

    async for _ in client._backfill_room("!room:example.org", "t0"):
        pass
    assert mock_nio_client.room_messages.call_count == 2


async def test_backfill_room_does_not_stop_on_empty_chunk_alone(client, mock_nio_client):
    # Empty chunk but end is still present — should continue
    resp1 = _make_room_messages_response([], end="page2")
    resp2 = _make_room_messages_response([], end=None)
    mock_nio_client.room_messages = AsyncMock(side_effect=[resp1, resp2])

    async for _ in client._backfill_room("!room:example.org", "t0"):
        pass
    assert mock_nio_client.room_messages.call_count == 2


async def test_backfill_room_respects_pages_max(client, mock_nio_client):
    # pages_max=2, always returns end token → should stop after 2 pages
    event = _make_text_event()
    mock_nio_client.room_messages = AsyncMock(
        return_value=_make_room_messages_response([event], end="always_more")
    )
    client._config.backfill_pages_max = 2
    client._embedding_client.embed_batch = AsyncMock(return_value=[[0.1] * 1536])

    async for _ in client._backfill_room("!room:example.org", "t0"):
        pass
    assert mock_nio_client.room_messages.call_count == 2


async def test_backfill_room_unlimited_when_pages_max_zero(client, mock_nio_client):
    # pages_max=0 → run until end is None
    events = [_make_text_event(event_id=f"$e{i}:x", body=f"msg{i}") for i in range(3)]
    side_effects = [
        _make_room_messages_response([events[0]], end="p2"),
        _make_room_messages_response([events[1]], end="p3"),
        _make_room_messages_response([events[2]], end=None),
    ]
    mock_nio_client.room_messages = AsyncMock(side_effect=side_effects)
    client._config.backfill_pages_max = 0
    client._embedding_client.embed_batch = AsyncMock(
        return_value=[[0.1] * 1536, [0.1] * 1536, [0.1] * 1536]
    )

    async for _ in client._backfill_room("!room:example.org", "t0"):
        pass
    assert mock_nio_client.room_messages.call_count == 3


# --- on_message callback ---

async def test_on_message_adds_to_buffer_and_indexes(client, mock_nio_client):
    room = MagicMock()
    room.room_id = "!room:example.org"
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)

    await client._on_message(room, event)

    assert len(client._buffer) == 1
    assert client._buffer[0].event_id == event.event_id
    client._vector_store.upsert.assert_called_once()


async def test_on_message_deduplicates_event_ids(client):
    room = MagicMock()
    room.room_id = "!room:example.org"
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)

    await client._on_message(room, event)
    await client._on_message(room, event)

    assert len(client._buffer) == 1
    assert client._vector_store.upsert.call_count == 1


# --- get_recent_messages ---

def _add_records(client, records):
    for r in records:
        client._buffer.append(r)


async def test_get_recent_messages_returns_last_k(client):
    for i in range(10):
        client._buffer.append(MessageRecord(f"$e{i}:x", "!r:x", "@a:x", "A", f"msg{i}", i))
    results = await client.get_recent_messages(k=3)
    assert len(results) == 3
    assert results[-1].body == "msg9"


async def test_get_recent_messages_filters_by_sender(client):
    client._buffer.append(MessageRecord("$1:x", "!r:x", "@alice:x", "Alice", "hi", 1))
    client._buffer.append(MessageRecord("$2:x", "!r:x", "@bob:x", "Bob", "hey", 2))
    results = await client.get_recent_messages(k=10, sender="@alice:x")
    assert all(r.sender == "@alice:x" for r in results)
    assert len(results) == 1


async def test_get_recent_messages_filters_by_room(client):
    client._buffer.append(MessageRecord("$1:x", "!room1:x", "@a:x", "A", "hi", 1))
    client._buffer.append(MessageRecord("$2:x", "!room2:x", "@a:x", "A", "hey", 2))
    results = await client.get_recent_messages(k=10, room_id="!room1:x")
    assert all(r.room_id == "!room1:x" for r in results)
    assert len(results) == 1


# --- send_message ---

async def test_send_message_calls_room_send(client, mock_nio_client):
    resp = MagicMock(spec=nio.RoomSendResponse)
    resp.event_id = "$sent:example.org"
    mock_nio_client.room_send = AsyncMock(return_value=resp)

    result = await client.send_message("!room:x", "Hello!")
    assert result["event_id"] == "$sent:example.org"
    mock_nio_client.room_send.assert_called_once_with(
        room_id="!room:x",
        message_type="m.room.message",
        content={"msgtype": "m.text", "body": "Hello!"},
    )
