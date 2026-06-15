import asyncio
import hashlib
import os
import pytest
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

from nio_mcp.matrix_client import MatrixMCPClient
from nio_mcp.models import MessageRecord
from nio_mcp.store import MessageStore
import nio


def _make_config(
    backfill_pages_max=2,
    backfill_limit=5,
    message_buffer_size=50,
    matrix_sync_timeout_ms=30000,
    ignored_room_ids=frozenset(),
):
    cfg = MagicMock()
    cfg.matrix_homeserver_url = "https://matrix.example.org"
    cfg.matrix_user_id = "@bot:example.org"
    cfg.matrix_device_id = "DEVID123"
    cfg.matrix_access_token = "syt_token"
    cfg.matrix_store_path = "/tmp/test_nio_store"
    cfg.matrix_key_backup_file = ""
    cfg.matrix_key_backup_passphrase = ""
    cfg.backfill_pages_max = backfill_pages_max
    cfg.backfill_limit = backfill_limit
    cfg.message_buffer_size = message_buffer_size
    cfg.matrix_sync_timeout_ms = matrix_sync_timeout_ms
    cfg.ignored_room_ids = ignored_room_ids
    return cfg


def _make_room_messages_response(chunk, end=None):
    resp = MagicMock(spec=nio.RoomMessagesResponse)
    resp.chunk = chunk
    resp.end = end
    return resp


def _make_room(room_id="!room:example.org", display_name="Test Room"):
    room = MagicMock()
    room.room_id = room_id
    room.display_name = display_name
    room.users = {}
    room.encrypted = False
    return room


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
def client(mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path):
    cfg = _make_config()
    cfg.matrix_store_path = str(tmp_path)
    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    c._client = mock_nio_client
    c._store.open()
    yield c
    c._store.close()


# --- start() behaviour ---

async def test_start_creates_store_dir_before_restore_login(
    mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    cfg = _make_config()
    cfg.matrix_store_path = str(tmp_path)
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
    c._store.close()

    assert call_order.index("makedirs") < call_order.index("restore_login")


async def test_start_calls_restore_login_with_env_credentials(
    mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    cfg = _make_config()
    cfg.matrix_store_path = str(tmp_path)
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
    c._store.close()

    mock_nio_client.restore_login.assert_called_once_with(
        user_id="@bot:example.org",
        device_id="DEVID123",
        access_token="syt_token",
    )


async def test_start_passes_initial_sync_next_batch_to_sync_forever(
    mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    cfg = _make_config()
    cfg.matrix_store_path = str(tmp_path)
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
    c._store.close()

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

    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    # Pre-populate the DB so start() takes the restart path and _restore_rooms_to_client runs.
    c._store.open()
    c._store.set_meta("backfill_complete", "1")
    c._store.upsert_room("!general:example.org", "General", encrypted=False)
    c._store.close()

    with patch("nio_mcp.matrix_client.asyncio.create_task") as create_task:
        def capture_and_close(coro):
            if not inspect.iscoroutine(coro):
                raise TypeError(f"Expected coroutine, got {type(coro)}")
            coro.close()
            return MagicMock()

        create_task.side_effect = capture_and_close
        await c.start()
    c._store.close()

    # On restart: skip initial sync and backfill entirely
    mock_nio_client.sync.assert_not_called()
    mock_nio_client.joined_rooms.assert_not_called()
    # sync_forever must be started from the stored token
    assert create_task.called
    mock_nio_client.sync_forever.assert_called_once_with(since="stored_t99", timeout=30000)


async def test_start_restart_with_persisted_room_does_not_crash_and_send_message_works(
    mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    """_restore_rooms_to_client must not set name/display_name on MatrixRoom (read-only
    property).  The room only needs to be present in client.rooms for send_message to work;
    the DB remains the source of truth for human-readable labels."""
    cfg = _make_config()
    cfg.matrix_store_path = str(tmp_path)
    mock_nio_client.loaded_sync_token = "stored_t99"

    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    c._store.open()
    c._store.set_meta("backfill_complete", "1")
    # Simulate a DM room whose display_name was derived from member names, not room.name.
    c._store.upsert_room("!dm:example.org", "Alice", encrypted=False)
    c._store.close()

    with patch("nio_mcp.matrix_client.asyncio.create_task") as create_task:
        create_task.side_effect = lambda coro: (coro.close(), MagicMock())[1]
        await c.start()  # must not raise AttributeError

    restored = mock_nio_client.rooms.get("!dm:example.org")
    assert restored is not None, "room must be in client.rooms after restart"
    assert restored.name is None, "name must stay unset; DB is source of truth for labels"

    # send_message must reach room_send without error
    resp = MagicMock(spec=nio.RoomSendResponse)
    resp.event_id = "$sent:x"
    mock_nio_client.room_send = AsyncMock(return_value=resp)
    result = await c.send_message("!dm:example.org", "hello")
    assert result["event_id"] == "$sent:x"

    c._store.close()


# --- backfill pagination ---

async def test_backfill_room_stops_when_end_is_none(client, mock_nio_client):
    event = _make_text_event()
    resp1 = _make_room_messages_response([event], end="page2")
    resp2 = _make_room_messages_response([_make_text_event("$b:x", body="B")], end=None)
    mock_nio_client.room_messages = AsyncMock(side_effect=[resp1, resp2])
    client._embedding_client.embed_batch = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])

    await client._backfill_room("!room:example.org", "t0")
    assert mock_nio_client.room_messages.call_count == 2


async def test_backfill_room_does_not_stop_on_empty_chunk_alone(client, mock_nio_client):
    # Empty chunk but end is still present — should continue
    resp1 = _make_room_messages_response([], end="page2")
    resp2 = _make_room_messages_response([], end=None)
    mock_nio_client.room_messages = AsyncMock(side_effect=[resp1, resp2])

    await client._backfill_room("!room:example.org", "t0")
    assert mock_nio_client.room_messages.call_count == 2


async def test_backfill_room_respects_pages_max(client, mock_nio_client):
    # pages_max=2, always returns end token → should stop after 2 pages
    event = _make_text_event()
    mock_nio_client.room_messages = AsyncMock(
        return_value=_make_room_messages_response([event], end="always_more")
    )
    client._config.backfill_pages_max = 2
    client._embedding_client.embed_batch = AsyncMock(return_value=[[0.1] * 1536])

    await client._backfill_room("!room:example.org", "t0")
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

    await client._backfill_room("!room:example.org", "t0")
    assert mock_nio_client.room_messages.call_count == 3


# --- on_message callback ---

async def test_on_message_adds_to_store_and_indexes(client, mock_nio_client):
    room = _make_room()
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)

    await client._on_message(room, event)

    messages = client._store.get_recent_messages(10)
    assert len(messages) == 1
    assert messages[0].event_id == event.event_id
    client._vector_store.upsert.assert_called_once()


async def test_on_message_deduplicates_event_ids(client):
    room = _make_room()
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)

    await client._on_message(room, event)
    await client._on_message(room, event)

    assert len(client._store.get_recent_messages(10)) == 1
    assert client._vector_store.upsert.call_count == 1


async def test_on_message_clears_pending_on_success(client):
    room = _make_room()
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)

    await client._on_message(room, event)

    assert not client._store.get_pending_messages()


async def test_on_message_retains_pending_on_index_failure(client):
    room = _make_room()
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(side_effect=RuntimeError("openai down"))

    await client._on_message(room, event)

    pending = client._store.get_pending_messages()
    assert any(r.event_id == event.event_id for r in pending)
    # message is in DB even though indexing failed
    assert len(client._store.get_recent_messages(10)) == 1


async def test_retry_pending_index_reindexes_and_clears(client):
    record = MessageRecord("$pend:x", "!r:x", "", "@a:x", "A", "lost msg", 1000)
    client._store.insert_message(record, indexed=False)
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)

    await client._retry_pending_index()

    client._vector_store.upsert.assert_called_once()
    client._webhook_dispatcher.dispatch.assert_not_called()
    assert not client._store.get_pending_messages()


async def test_retry_pending_index_is_idempotent(client):
    record = MessageRecord("$pend:x", "!r:x", "", "@a:x", "A", "lost msg", 1000)
    client._store.insert_message(record, indexed=False)
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)

    await client._retry_pending_index()
    await client._retry_pending_index()  # second call finds nothing pending

    client._vector_store.upsert.assert_called_once()


async def test_retry_pending_index_leaves_failures_pending(client):
    record = MessageRecord("$pend:x", "!r:x", "", "@a:x", "A", "lost msg", 1000)
    client._store.insert_message(record, indexed=False)
    client._embedding_client.embed = AsyncMock(side_effect=RuntimeError("qdrant down"))

    await client._retry_pending_index()

    pending = client._store.get_pending_messages()
    assert any(r.event_id == record.event_id for r in pending)


async def test_on_message_retains_pending_on_webhook_failure(client):
    """Webhook failure must keep the record in the pending state for retry."""
    room = _make_room()
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)
    client._webhook_dispatcher.dispatch = AsyncMock(side_effect=RuntimeError("webhook down"))

    await client._on_message(room, event)

    pending = client._store.get_pending_messages()
    assert any(r.event_id == event.event_id for r in pending)
    # Qdrant upsert was still attempted before webhook
    client._vector_store.upsert.assert_called_once()


async def test_on_message_defers_indexing_when_db_write_fails(client):
    """If the DB write fails, indexing must not run."""
    room = _make_room()
    event = _make_text_event()
    client._store.insert_message = MagicMock(side_effect=OSError("disk full"))

    await client._on_message(room, event)

    client._vector_store.upsert.assert_not_called()


async def test_index_message_embeds_body_only(client):
    record = MessageRecord("$body:x", "!r:x", "", "@alice:x", "Alice", "hello world", 1000)
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)

    await client._index_message(record)

    client._embedding_client.embed.assert_called_once_with("hello world")
    client._webhook_dispatcher.dispatch.assert_called_once_with(record)


async def test_index_message_without_webhook_dispatch(client):
    record = MessageRecord("$body:x", "!r:x", "", "@alice:x", "Alice", "hello world", 1000)
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)

    await client._index_message(record, dispatch_webhook=False)

    client._embedding_client.embed.assert_called_once_with("hello world")
    client._webhook_dispatcher.dispatch.assert_not_called()


async def test_batch_index_embeds_bodies_only(client):
    records = [
        MessageRecord("$1:x", "!r:x", "", "@alice:x", "Alice", "first body", 1000),
        MessageRecord("$2:x", "!r:x", "", "@bob:x", "Bob", "second body", 1001),
    ]
    client._embedding_client.embed_batch = AsyncMock(
        return_value=[[0.1] * 1536, [0.2] * 1536]
    )

    await client._batch_index(records)

    client._embedding_client.embed_batch.assert_called_once_with(["first body", "second body"])


async def test_batch_index_failure_leaves_records_pending(client):
    records = [
        MessageRecord("$fail1:x", "!r:x", "", "@alice:x", "Alice", "msg1", 1000),
        MessageRecord("$fail2:x", "!r:x", "", "@bob:x", "Bob", "msg2", 1001),
    ]
    client._embedding_client.embed_batch = AsyncMock(side_effect=RuntimeError("openai down"))

    await client._batch_index(records)

    pending_ids = {r.event_id for r in client._store.get_pending_messages()}
    assert "$fail1:x" in pending_ids
    assert "$fail2:x" in pending_ids


# --- get_recent_messages ---

async def test_get_recent_messages_returns_last_k(client):
    for i in range(10):
        client._store.insert_message(
            MessageRecord(f"$e{i}:x", "!r:x", "", "@a:x", "A", f"msg{i}", i),
            indexed=True,
        )
    results = await client.get_recent_messages(k=3)
    assert len(results) == 3
    assert results[-1].body == "msg9"


async def test_get_recent_messages_filters_by_sender(client):
    client._store.insert_message(
        MessageRecord("$1:x", "!r:x", "", "@alice:x", "Alice", "hi", 1), indexed=True
    )
    client._store.insert_message(
        MessageRecord("$2:x", "!r:x", "", "@bob:x", "Bob", "hey", 2), indexed=True
    )
    results = await client.get_recent_messages(k=10, sender="@alice:x")
    assert all(r.sender == "@alice:x" for r in results)
    assert len(results) == 1


async def test_get_recent_messages_filters_by_room(client):
    client._store.insert_message(
        MessageRecord("$1:x", "!room1:x", "", "@a:x", "A", "hi", 1), indexed=True
    )
    client._store.insert_message(
        MessageRecord("$2:x", "!room2:x", "", "@a:x", "A", "hey", 2), indexed=True
    )
    results = await client.get_recent_messages(k=10, room_id="!room1:x")
    assert all(r.room_id == "!room1:x" for r in results)
    assert len(results) == 1


# --- get_room_info ---

def test_get_room_info_returns_name_and_members(client, mock_nio_client):
    client._store.upsert_room("!room:example.org", "General")
    client._store.upsert_member("!room:example.org", "@alice:example.org", "Alice")
    client._store.upsert_member("!room:example.org", "@bob:example.org", "Bob")

    result = client.get_room_info("!room:example.org")

    assert result["room_id"] == "!room:example.org"
    assert result["name"] == "General"
    assert len(result["members"]) == 2
    mxids = {m["mxid"] for m in result["members"]}
    assert mxids == {"@alice:example.org", "@bob:example.org"}
    names = {m["display_name"] for m in result["members"]}
    assert names == {"Alice", "Bob"}


def test_get_room_info_falls_back_to_localpart_when_display_name_missing(client, mock_nio_client):
    client._store.upsert_room("!q:example.org", "Quiet Room")
    client._store.upsert_member("!q:example.org", "@carol:example.org", "carol")

    result = client.get_room_info("!q:example.org")

    assert result["members"][0]["mxid"] == "@carol:example.org"
    assert result["members"][0]["display_name"] == "carol"


def test_get_room_info_returns_error_for_unknown_room(client, mock_nio_client):
    result = client.get_room_info("!unknown:example.org")
    assert "error" in result


def test_get_room_info_raises_when_client_not_started(client):
    client._client = None
    with pytest.raises(RuntimeError, match="Client not started"):
        client.get_room_info("!room:example.org")


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
                        mock_nio_client, key_file="", passphrase="secret"):
    cfg = _make_config()
    cfg.matrix_store_path = str(tmp_path)
    cfg.matrix_key_backup_file = key_file
    cfg.matrix_key_backup_passphrase = passphrase
    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    c._store.open()
    c._client = mock_nio_client
    return c


def _write_key_file(tmp_path, content=KEY_CONTENT) -> str:
    key_file = tmp_path / "keys.txt"
    key_file.write_text(content)
    return str(key_file)


def _fingerprint(key_file: str, passphrase: str = "secret") -> str:
    h = hashlib.sha256()
    with open(key_file, "rb") as f:
        h.update(f.read())
    h.update(passphrase.encode())
    return h.hexdigest()


async def test_import_key_backup_returns_false_when_file_empty(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client, key_file="")
    result = await c._import_key_backup()
    assert result is False
    mock_nio_client.import_keys.assert_not_called()
    c._store.close()


async def test_import_key_backup_returns_false_when_sentinel_matches(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    key_file = _write_key_file(tmp_path)
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client, key_file=key_file)
    c._store.set_meta("key_backup_imported", _fingerprint(key_file))
    result = await c._import_key_backup()
    assert result is False
    mock_nio_client.import_keys.assert_not_called()
    c._store.close()


async def test_import_key_backup_returns_true_and_calls_import(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    key_file = _write_key_file(tmp_path)
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client, key_file=key_file)
    result = await c._import_key_backup()
    assert result is True
    mock_nio_client.import_keys.assert_called_once_with(key_file, "secret")
    c._store.close()


async def test_import_key_backup_writes_fingerprint_sentinel_on_success(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    key_file = _write_key_file(tmp_path)
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client, key_file=key_file)
    await c._import_key_backup()
    assert c._store.get_meta("key_backup_imported") == _fingerprint(key_file)
    c._store.close()


async def test_import_key_backup_skips_sentinel_on_error(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    key_file = _write_key_file(tmp_path)
    mock_nio_client.import_keys = AsyncMock(side_effect=RuntimeError("bad passphrase"))
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client, key_file=key_file)
    with pytest.raises(RuntimeError, match="bad passphrase"):
        await c._import_key_backup()
    assert c._store.get_meta("key_backup_imported") is None
    c._store.close()


async def test_import_key_backup_fails_if_file_missing(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client, key_file="/nonexistent/keys.txt")
    with pytest.raises(FileNotFoundError, match="MATRIX_KEY_BACKUP_FILE"):
        await c._import_key_backup()
    mock_nio_client.import_keys.assert_not_called()
    c._store.close()


async def test_import_key_backup_fails_if_sentinel_fingerprint_mismatch(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    key_file = _write_key_file(tmp_path)
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client, key_file=key_file, passphrase="new-pass")
    c._store.set_meta("key_backup_imported", _fingerprint(key_file, passphrase="old-pass"))
    with pytest.raises(RuntimeError, match="MATRIX_KEY_BACKUP_FILE or MATRIX_KEY_BACKUP_PASSPHRASE has changed"):
        await c._import_key_backup()
    mock_nio_client.import_keys.assert_not_called()
    c._store.close()


async def test_import_key_backup_fails_if_sentinel_empty(
    mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    key_file = _write_key_file(tmp_path)
    c = _make_backup_client(tmp_path, vector_store, embedding_client, webhook_dispatcher,
                            mock_nio_client, key_file=key_file)
    c._store.set_meta("key_backup_imported", "")
    with pytest.raises(RuntimeError, match="MATRIX_KEY_BACKUP_FILE or MATRIX_KEY_BACKUP_PASSPHRASE has changed"):
        await c._import_key_backup()
    mock_nio_client.import_keys.assert_not_called()
    c._store.close()


# --- start(): backfill sentinel cleared after first key import ---

def _make_start_helpers(tmp_path, mock_nio_client, vector_store, embedding_client,
                        webhook_dispatcher, key_file=""):
    """Return a client configured for start() tests with a key backup."""
    cfg = _make_config()
    cfg.matrix_store_path = str(tmp_path)
    cfg.matrix_key_backup_file = key_file
    cfg.matrix_key_backup_passphrase = "secret" if key_file else ""
    c = MatrixMCPClient(cfg, vector_store, embedding_client, webhook_dispatcher)
    mock_nio_client.loaded_sync_token = "stored_t99"
    mock_nio_client.sync.return_value = _make_initial_sync()
    mock_nio_client.joined_rooms.return_value = MagicMock(rooms=[])
    return c


async def test_start_clears_backfill_sentinel_when_key_first_imported(
    mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    """When a key backup is imported for the first time and a completed backfill exists,
    start() must clear the backfill sentinel so backfill re-runs and decrypts history."""
    key_file = _write_key_file(tmp_path)
    c = _make_start_helpers(tmp_path, mock_nio_client, vector_store, embedding_client,
                            webhook_dispatcher, key_file=key_file)
    # Backfill already done, but no key imported yet — first run with a key.
    c._store.open()
    c._store.set_meta("backfill_complete", "1")
    c._store.close()

    with patch("nio_mcp.matrix_client.asyncio.create_task") as create_task:
        create_task.side_effect = lambda coro: (coro.close(), MagicMock())[1]
        await c.start()
    c._store.close()

    # Backfill must have run (sync + joined_rooms called), not the fast restart path.
    mock_nio_client.sync.assert_called_once()
    mock_nio_client.joined_rooms.assert_called_once()


async def test_start_does_not_clear_backfill_sentinel_for_same_key(
    mock_makedirs, mock_nio_client, vector_store, embedding_client, webhook_dispatcher, tmp_path
):
    """Re-running with the same key backup (sentinel fingerprint matches) must NOT
    trigger a re-backfill — start() should take the fast restart path."""
    key_file = _write_key_file(tmp_path)
    c = _make_start_helpers(tmp_path, mock_nio_client, vector_store, embedding_client,
                            webhook_dispatcher, key_file=key_file)
    # Both sentinels already in DB — normal restart state.
    c._store.open()
    c._store.set_meta("backfill_complete", "1")
    c._store.set_meta("key_backup_imported", _fingerprint(key_file))
    c._store.close()

    with patch("nio_mcp.matrix_client.asyncio.create_task") as create_task:
        create_task.side_effect = lambda coro: (coro.close(), MagicMock())[1]
        await c.start()
    c._store.close()

    # Fast restart path: no sync, no backfill.
    mock_nio_client.sync.assert_not_called()
    mock_nio_client.joined_rooms.assert_not_called()
    mock_nio_client.sync_forever.assert_called_once_with(since="stored_t99", timeout=30000)


# --- ignored rooms ---

def _make_two_room_sync(room_ids: list[str], prev_batch: str = "t1") -> MagicMock:
    """Build a SyncResponse mock with a timeline entry for each room_id."""
    sync = MagicMock(spec=nio.SyncResponse)
    sync.next_batch = "next_batch_token"
    join = {}
    for rid in room_ids:
        timeline = MagicMock()
        timeline.prev_batch = prev_batch
        room_info = MagicMock()
        room_info.timeline = timeline
        join[rid] = room_info
    sync.rooms = MagicMock()
    sync.rooms.join = join
    return sync


async def test_on_message_skips_ignored_room(client, mock_nio_client):
    IGNORED = "!ignored:example.org"
    client._config.ignored_room_ids = frozenset([IGNORED])
    room = _make_room(room_id=IGNORED)
    event = _make_text_event()

    await client._on_message(room, event)

    assert client._store.get_recent_messages(10) == []
    client._vector_store.upsert.assert_not_called()


async def test_on_message_allows_non_ignored_room(client, mock_nio_client):
    client._config.ignored_room_ids = frozenset(["!other:example.org"])
    room = _make_room(room_id="!allowed:example.org")
    event = _make_text_event()
    client._embedding_client.embed = AsyncMock(return_value=[0.1] * 1536)

    await client._on_message(room, event)

    assert len(client._store.get_recent_messages(10)) == 1
    client._vector_store.upsert.assert_called_once()


async def test_backfill_skips_ignored_rooms(client, mock_nio_client):
    IGNORED = "!ignored:example.org"
    REAL = "!real:example.org"
    client._config.ignored_room_ids = frozenset([IGNORED])

    rooms_resp = MagicMock()
    rooms_resp.rooms = [IGNORED, REAL]
    mock_nio_client.joined_rooms = AsyncMock(return_value=rooms_resp)
    mock_nio_client.room_messages = AsyncMock(
        return_value=_make_room_messages_response([], end=None)
    )

    await client._backfill(_make_two_room_sync([IGNORED, REAL]))

    assert mock_nio_client.room_messages.call_count == 1
    assert mock_nio_client.room_messages.call_args.kwargs["room_id"] == REAL


async def test_backfill_includes_all_when_ignored_rooms_empty(client, mock_nio_client):
    ROOM1 = "!room1:example.org"
    ROOM2 = "!room2:example.org"
    client._config.ignored_room_ids = frozenset()

    rooms_resp = MagicMock()
    rooms_resp.rooms = [ROOM1, ROOM2]
    mock_nio_client.joined_rooms = AsyncMock(return_value=rooms_resp)
    mock_nio_client.room_messages = AsyncMock(
        return_value=_make_room_messages_response([], end=None)
    )

    await client._backfill(_make_two_room_sync([ROOM1, ROOM2]))

    assert mock_nio_client.room_messages.call_count == 2


async def test_index_initial_sync_skips_ignored_rooms(client, mock_nio_client):
    IGNORED = "!ignored:example.org"
    REAL = "!real:example.org"
    client._config.ignored_room_ids = frozenset([IGNORED])

    ignored_event = _make_text_event("$ign:x", body="should not index")
    real_event = _make_text_event("$real:x", body="should index")

    sync = MagicMock(spec=nio.SyncResponse)
    def _room_info(event):
        ri = MagicMock()
        ri.timeline = MagicMock()
        ri.timeline.events = [event]
        return ri

    sync.rooms = MagicMock()
    sync.rooms.join = {IGNORED: _room_info(ignored_event), REAL: _room_info(real_event)}

    client._embedding_client.embed_batch = AsyncMock(return_value=[[0.1] * 1536])

    await client._index_initial_sync(sync)

    # Only the real room's message should have been indexed
    assert client._embedding_client.embed_batch.call_count == 1
    indexed_bodies = client._embedding_client.embed_batch.call_args.args[0]
    assert "should index" in indexed_bodies
    assert "should not index" not in indexed_bodies
