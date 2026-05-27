import asyncio
import os
import tempfile
import pytest
import inspect
from unittest.mock import AsyncMock, MagicMock, patch, call
from collections import deque

from nio_mcp.matrix_client import MatrixMCPClient
from nio_mcp.models import MessageRecord
import nio


def _make_config(
    backfill_pages_max=2,
    backfill_limit=5,
    message_buffer_size=50,
    matrix_sync_timeout_ms=30000,
):
    cfg = MagicMock()
    cfg.matrix_homeserver_url = "https://matrix.example.org"
    cfg.matrix_user_id = "@bot:example.org"
    cfg.matrix_device_id = "DEVID123"
    cfg.matrix_access_token = "syt_token"
    cfg.matrix_store_path = "/tmp/test_nio_store"
    cfg.matrix_key_backup_content = ""
    cfg.matrix_key_backup_passphrase = ""
    cfg.backfill_pages_max = backfill_pages_max
    cfg.backfill_limit = backfill_limit
    cfg.message_buffer_size = message_buffer_size
    cfg.matrix_sync_timeout_ms = matrix_sync_timeout_ms
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
        instance = MagicMock(spec=nio.AsyncClient)
        cls.return_value = instance
        instance.restore_login = MagicMock()
        instance.add_event_callback = MagicMock()
        instance.sync = AsyncMock()
        instance.joined_rooms = AsyncMock()
        instance.room_messages = AsyncMock()
        instance.room_context = AsyncMock()
        instance.room_send = AsyncMock()
        instance.close = AsyncMock()
        instance.sync_forever = AsyncMock()
        instance.import_keys = AsyncMock()
        instance.rooms = {}
        instance.loaded_sync_token = ""
        yield instance


@pytest.fixture
def mock_makedirs():
    with patch("nio_mcp.matrix_client.os.makedirs") as m:
        yield m


@pytest.fixture
def vector_store():
    from nio_mcp.vector_store import VectorStore
    vs = MagicMock(spec=VectorStore)
    vs.upsert = AsyncMock()
    vs.search = AsyncMock()
    vs.scroll = AsyncMock()
    vs.init_collection = AsyncMock()
    vs.close = AsyncMock()
    return vs


@pytest.fixture
def embedding_client():
    from nio_mcp.embeddings import EmbeddingClient
    ec = MagicMock(spec=EmbeddingClient)
    ec.embed = AsyncMock(return_value=[0.1] * 1536)
    ec.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
    ec.close = AsyncMock()
    return ec


@pytest.fixture
def webhook_dispatcher():
    from nio_mcp.webhook import WebhookDispatcher
    wd = MagicMock(spec=WebhookDispatcher)
    wd.dispatch = AsyncMock()
    wd.start = AsyncMock()
    wd.close = AsyncMock()
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
    with patch("nio_mcp.matrix_client.asyncio.create_task") as mock_create:
        def _check_and_close(coro):
            if not inspect.iscoroutine(coro):
                raise TypeError(f"Expected coroutine, got {type(coro)}")
            coro.close()
            return MagicMock()
        mock_create.side_effect = _check_and_close
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
    with patch("nio_mcp.matrix_client.asyncio.create_task") as mock_create:
        def _check_and_close(coro):
            if not inspect.iscoroutine(coro):
                raise TypeError(f"Expected coroutine, got {type(coro)}")
            coro.close()
            return MagicMock()
        mock_create.side_effect = _check_and_close
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

    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    with patch("nio_mcp.matrix_client.asyncio.create_task") as create_task:
        def close_coro(coro):
            if not inspect.iscoroutine(coro):
                raise TypeError(f"Expected coroutine, got {type(coro)}")
            coro.close()
            return MagicMock()

        create_task.side_effect = close_coro
        await c.start()

    mock_nio_client.sync_forever.assert_called_once_with(
        since=initial_sync.next_batch,
        timeout=30000,
    )


async def test_start_resumes_from_stored_token_on_restart(
    mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    cfg = _make_config()
    cfg.matrix_store_path = str(tmp_path)
    mock_nio_client.loaded_sync_token = "stored_t99"

    # Create the sentinel so start() takes the restart path.
    (tmp_path / "backfill_complete").write_text("")

    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    with patch("nio_mcp.matrix_client.asyncio.create_task") as create_task:
        def capture_and_close(coro):
            if not inspect.iscoroutine(coro):
                raise TypeError(f"Expected coroutine, got {type(coro)}")
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
    client._save_pending_index = MagicMock()

    await client._on_message(room, event)

    assert len(client._buffer) == 1
    assert client._buffer[0].event_id == event.event_id
    client._vector_store.upsert.assert_called_once()


async def test_on_message_deduplicates_event_ids(client):
    room = MagicMock()
    room.room_id = "!room:example.org"
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)
    client._save_pending_index = MagicMock()

    await client._on_message(room, event)
    await client._on_message(room, event)

    assert len(client._buffer) == 1
    assert client._vector_store.upsert.call_count == 1


async def test_on_message_clears_pending_on_success(client):
    room = MagicMock()
    room.room_id = "!room:example.org"
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)
    client._save_pending_index = MagicMock()

    await client._on_message(room, event)

    assert event.event_id not in client._pending_index


async def test_on_message_retains_pending_on_index_failure(client):
    room = MagicMock()
    room.room_id = "!room:example.org"
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(side_effect=RuntimeError("openai down"))
    client._save_pending_index = MagicMock()

    await client._on_message(room, event)

    assert event.event_id in client._pending_index
    # message is still in buffer even though indexing failed
    assert len(client._buffer) == 1


async def test_retry_pending_index_reindexes_and_clears(client):
    record = MessageRecord("$pend:x", "!r:x", "@a:x", "A", "lost msg", 1000)
    client._pending_index = {record.event_id: record}
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)
    client._save_pending_index = MagicMock()

    await client._retry_pending_index()

    client._vector_store.upsert.assert_called_once()
    assert record.event_id not in client._pending_index
    assert record.event_id in client._seen_event_ids
    assert any(r.event_id == record.event_id for r in client._buffer)


async def test_retry_pending_index_skips_already_seen(client):
    record = MessageRecord("$pend:x", "!r:x", "@a:x", "A", "lost msg", 1000)
    client._pending_index = {record.event_id: record}
    client._seen_event_ids.add(record.event_id)
    client._buffer.append(record)
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)
    client._save_pending_index = MagicMock()

    await client._retry_pending_index()

    # should index exactly once; buffer should not have a duplicate
    client._vector_store.upsert.assert_called_once()
    assert len([r for r in client._buffer if r.event_id == record.event_id]) == 1


async def test_retry_pending_index_leaves_failures_pending(client):
    record = MessageRecord("$pend:x", "!r:x", "@a:x", "A", "lost msg", 1000)
    client._pending_index = {record.event_id: record}
    client._embedding_client.embed = AsyncMock(side_effect=RuntimeError("qdrant down"))
    client._save_pending_index = MagicMock()

    await client._retry_pending_index()

    assert record.event_id in client._pending_index


async def test_on_message_retains_pending_on_webhook_failure(client):
    """Webhook failure must keep the record in the pending journal for retry."""
    room = MagicMock()
    room.room_id = "!room:example.org"
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)
    client._webhook_dispatcher.dispatch = AsyncMock(side_effect=RuntimeError("webhook down"))
    client._save_pending_index = MagicMock()

    await client._on_message(room, event)

    assert event.event_id in client._pending_index
    # Qdrant upsert was still attempted before webhook
    client._vector_store.upsert.assert_called_once()


async def test_on_message_defers_indexing_when_pre_save_fails(client):
    """If the pre-index journal write fails, indexing must not run.

    The record stays in _pending_index so the next successful write for any
    event carries it to disk, giving it a restart-recovery path.
    """
    room = MagicMock()
    room.room_id = "!room:example.org"
    event = _make_text_event()
    client._save_pending_index = MagicMock(side_effect=OSError("disk full"))

    await client._on_message(room, event)

    client._vector_store.upsert.assert_not_called()
    assert event.event_id in client._pending_index


async def test_index_message_embeds_body_only(client):
    record = MessageRecord("$body:x", "!r:x", "@alice:x", "Alice", "hello world", 1000)
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)

    await client._index_message(record)

    client._embedding_client.embed.assert_called_once_with("hello world")


async def test_batch_index_embeds_bodies_only(client):
    records = [
        MessageRecord("$1:x", "!r:x", "@alice:x", "Alice", "first body", 1000),
        MessageRecord("$2:x", "!r:x", "@bob:x", "Bob", "second body", 1001),
    ]
    client._embedding_client.embed_batch = AsyncMock(
        return_value=[[0.1] * 1536, [0.2] * 1536]
    )

    await client._batch_index(records)

    client._embedding_client.embed_batch.assert_called_once_with(["first body", "second body"])


async def test_batch_index_failure_adds_records_to_pending_index(client):
    records = [
        MessageRecord("$fail1:x", "!r:x", "@alice:x", "Alice", "msg1", 1000),
        MessageRecord("$fail2:x", "!r:x", "@bob:x", "Bob", "msg2", 1001),
    ]
    client._embedding_client.embed_batch = AsyncMock(side_effect=RuntimeError("openai down"))
    client._save_pending_index = MagicMock()

    await client._batch_index(records)

    assert "$fail1:x" in client._pending_index
    assert "$fail2:x" in client._pending_index
    client._save_pending_index.assert_called()


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


# --- _import_key_backup ---

KEY_CONTENT = "-----BEGIN MEGOLM SESSION DATA-----\nABCDEF==\n-----END MEGOLM SESSION DATA-----"


def _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                        mock_nio_client, content=KEY_CONTENT, passphrase="secret"):
    cfg = _make_config()
    cfg.matrix_store_path = str(tmp_path)
    cfg.matrix_key_backup_content = content
    cfg.matrix_key_backup_passphrase = passphrase
    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    c._client = mock_nio_client
    return c


async def test_import_key_backup_noop_when_content_empty(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client, content="")
    await c._import_key_backup()
    mock_nio_client.import_keys.assert_not_called()


async def test_import_key_backup_noop_when_sentinel_exists(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    (tmp_path / "key_backup_imported").write_text("")
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client)
    await c._import_key_backup()
    mock_nio_client.import_keys.assert_not_called()


async def test_import_key_backup_passes_content_and_passphrase(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    # Use a real async function so we can read the temp file while it still exists.
    captured = {}
    async def capture(path, passphrase):
        with open(path) as f:
            captured["content"] = f.read()
        captured["passphrase"] = passphrase

    mock_nio_client.import_keys = capture
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client)
    await c._import_key_backup()

    assert captured["content"] == KEY_CONTENT
    assert captured["passphrase"] == "secret"


async def test_import_key_backup_deletes_temp_file_after_success(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    created_paths = []
    real_mkstemp = tempfile.mkstemp

    def capturing_mkstemp(**kwargs):
        fd, path = real_mkstemp(**kwargs)
        created_paths.append(path)
        return fd, path

    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client)
    with patch("nio_mcp.matrix_client.tempfile.mkstemp", side_effect=capturing_mkstemp):
        await c._import_key_backup()

    assert created_paths, "mkstemp was never called"
    assert not os.path.exists(created_paths[0])


async def test_import_key_backup_writes_sentinel_on_success(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client)
    await c._import_key_backup()
    assert (tmp_path / "key_backup_imported").exists()


async def test_import_key_backup_deletes_temp_file_and_skips_sentinel_on_error(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    mock_nio_client.import_keys = AsyncMock(side_effect=RuntimeError("bad passphrase"))
    created_paths = []
    real_mkstemp = tempfile.mkstemp

    def capturing_mkstemp(**kwargs):
        fd, path = real_mkstemp(**kwargs)
        created_paths.append(path)
        return fd, path

    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client)
    with patch("nio_mcp.matrix_client.tempfile.mkstemp", side_effect=capturing_mkstemp):
        with pytest.raises(RuntimeError, match="bad passphrase"):
            await c._import_key_backup()

    assert not os.path.exists(created_paths[0])
    assert not (tmp_path / "key_backup_imported").exists()
