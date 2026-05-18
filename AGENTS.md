# AGENTS.md — Developer Context for nio-mcp

This file captures the context needed to work on this codebase without having to re-read external documentation. It covers the project's internal design, the parts of the matrix-nio API that are actually used, and the non-obvious decisions baked into the implementation.

---

## Repository layout

```
src/nio_mcp/
├── config.py         # Pydantic Settings — all env vars in one place
├── models.py         # MessageRecord, SearchResult — the two data shapes used everywhere
├── embeddings.py     # Thin async wrapper around openai.AsyncOpenAI embeddings
├── vector_store.py   # Thin async wrapper around qdrant_client.AsyncQdrantClient
├── matrix_client.py  # The bulk of the logic: Matrix session, backfill, live sync, buffer
├── webhook.py        # HTTP POST dispatcher + per-subscriber SSE queue fan-out
└── server.py         # MCP tool definitions + FastAPI SSE app; wires everything together

tests/
├── unit/             # All external I/O mocked with pytest-mock / respx
└── integration/      # Real Qdrant required; auto-skipped if not reachable
```

Entry point: `server.py:main()` runs an `anyio` task group with two concurrent tasks — the MCP stdio server and a uvicorn/FastAPI server for SSE.

---

## Component wiring

`server.py` constructs all components and holds module-level references to `_matrix_client` and `_webhook_dispatcher` (used by MCP tool handlers and the SSE endpoint respectively). The dependency graph is:

```
MatrixMCPClient
  ├── EmbeddingClient    (called during backfill and on each live message)
  ├── VectorStore        (upsert during indexing, search during search_messages tool)
  └── WebhookDispatcher  (dispatch() called on each indexed live message)
```

`search_messages` is the one MCP tool that constructs its own `EmbeddingClient` and `VectorStore` per call rather than reusing the shared instances. This is intentional simplicity — searches are infrequent and the clients are stateless.

---

## matrix-nio API used in this project

### `AsyncClient` construction

```python
from nio import AsyncClient, ClientConfig

client = AsyncClient(
    homeserver="https://matrix.example.org",
    user="@bot:example.org",
    store_path="/data/nio_store",       # directory for Olm SQLite DB
    config=ClientConfig(store_sync_tokens=True),
)
```

`store_path` must be an **existing directory** — nio calls `load_store()` internally which opens a SQLite file there. Create it with `os.makedirs(path, exist_ok=True)` before constructing the client. `ClientConfig(store_sync_tokens=True)` persists the sync token to the store so restarts resume where they left off.

### `restore_login()`

```python
client.restore_login(
    user_id="@bot:example.org",
    device_id="DEVID123",
    access_token="syt_...",
)
```

Sets credentials on the client object directly. Does **not** require a prior persisted session — it works from env vars on a fresh deployment. The distinction from `login(password=...)` is that `restore_login` skips the authentication round-trip and reuses an existing token. `device_id` must be stable across restarts for E2EE; without it the Olm device identity cannot be reconstructed.

### `sync(full_state=True)` → `SyncResponse`

```python
initial_sync: SyncResponse = await client.sync(full_state=True)
```

Returns the current server state. Key fields used here:

| Field | Type | Description |
|---|---|---|
| `next_batch` | `str` | Sync token — pass to `sync_forever(since=...)` to avoid replaying events |
| `rooms.join` | `dict[str, JoinedRoomSyncResponse]` | Keyed by room ID |
| `rooms.join[room_id].timeline.events` | `list[Event]` | The events included in this sync's window — **must be indexed explicitly** before starting `sync_forever`, as they fall between the backfill's `prev_batch` and the live sync's `next_batch` |
| `rooms.join[room_id].timeline.prev_batch` | `str \| None` | Pagination token pointing to just before this sync's timeline slice — used as the `start` for `room_messages()` backfill |

`full_state=True` ensures `rooms.join` is populated even for rooms with no events in this sync window. It also causes nio to populate `client.rooms[room_id].users` with the full member list including display names — this is relied on by `_resolve_display_name()` during backfill and initial-sync indexing to look up human-readable sender names.

### `sync_forever(since=..., timeout=30000)`

```python
self._sync_task = asyncio.create_task(
    client.sync_forever(since=initial_sync.next_batch, timeout=30000)
)
```

Long-polls the server and fires registered event callbacks for each new event. `timeout` is the server-side long-poll timeout in milliseconds. Always pass `since=` to avoid receiving events already seen during backfill. Run as a background `asyncio.Task`; cancel it in `stop()` to shut down cleanly.

### `add_event_callback(callback, filter)`

```python
client.add_event_callback(self._on_message, RoomMessageText)
```

Registers an async callback invoked for each timeline event matching the filter type. Must be registered **before** `sync_forever` starts — callbacks registered after will miss events already delivered. The callback signature is:

```python
async def callback(room: MatrixRoom, event: RoomMessageText) -> None
```

`room.room_id` gives the room ID. `event.event_id`, `event.sender`, `event.body`, `event.server_timestamp` are the fields used here. `room.users` is a `dict[str, RoomMember]` keyed by Matrix ID; `RoomMember.display_name` gives the human-readable name used when constructing the embedding text.

### `room_messages()` — paginated history

```python
response = await client.room_messages(
    room_id="!abc:example.org",
    start=prev_batch_token,         # pagination cursor; None = current end of timeline
    direction=nio.MessageDirection.back,  # walk backwards (older messages)
    limit=100,
)
```

`RoomMessagesResponse` fields:

| Field | Type | Description |
|---|---|---|
| `chunk` | `list[Event]` | Events in this page (mixed types — filter for `RoomMessageText`) |
| `end` | `str \| None` | Cursor for the next page. **`None` means end of history** — this is the correct termination condition per the Matrix spec. An empty `chunk` does not mean pagination is finished. |

**Pagination pattern used in `_backfill_room()`:**

```python
while pages_max == 0 or page < pages_max:
    response = await client.room_messages(room_id=room_id, start=prev_batch, ...)
    # process response.chunk ...
    if response.end is None:   # spec-correct stop: absence of end token
        break
    prev_batch = response.end
    page += 1
```

`BACKFILL_PAGES_MAX=0` means unlimited. The backfill `start` token comes from `initial_sync.rooms.join[room_id].timeline.prev_batch`, which anchors pagination to the point just before the initial sync — preventing overlap with events that `sync_forever` will deliver.

### `room_context(room_id, event_id, limit)`

```python
response = await client.room_context(
    room_id="!abc:example.org",
    event_id="$found:example.org",
    limit=10,   # total events before + after
)
```

`RoomContextResponse` fields used: `response.event` (the target event), `response.events_before` (list), `response.events_after` (list). Returns `nio.ErrorResponse` on failure — check with `isinstance(response, nio.ErrorResponse)`.

### `room_send()`

```python
response = await client.room_send(
    room_id="!abc:example.org",
    message_type="m.room.message",
    content={"msgtype": "m.text", "body": "Hello!"},
)
```

Returns `nio.RoomSendResponse` (has `.event_id`) on success, `nio.ErrorResponse` on failure. The distinction matters — check with `isinstance`.

### `joined_rooms()` → `JoinedRoomsResponse`

```python
resp = await client.joined_rooms()
room_ids = resp.rooms   # list[str]
```

Returns the list of room IDs the bot has joined. Called once during backfill.

### Error handling pattern

nio methods return either a success response object or a `nio.ErrorResponse`. Never raises on Matrix-level errors. Always check the return type:

```python
if isinstance(response, nio.ErrorResponse):
    logger.warning("Failed: %s", response)
    return
```

---

## Startup sequence in detail

Understanding this sequence is critical for any change to `matrix_client.py:start()`. There are two distinct paths depending on whether a sync token was persisted from a previous run.

### Restart path (stored token present)

`client.loaded_sync_token` is populated by nio after `restore_login()` when `store_sync_tokens=True` and a prior sync has been persisted to the store. When it is non-empty, `start()` skips the initial sync and backfill entirely:

```
1. os.makedirs + AsyncClient + restore_login  (same as fresh start)
       ↓ loaded_sync_token is now set from the store
2. client.add_event_callback(_on_message, RoomMessageText)
3. asyncio.create_task(client.sync_forever(since=loaded_sync_token, ...))
       ↓ Matrix server delivers all events missed during downtime from that token forward
```

No backfill is needed — `sync_forever` catches up from the stored position. If the token is too old and the server has expired it, the sync will fail (not currently handled; treat as a fresh-start edge case).

### Fresh-start path (no stored token)

```
1. os.makedirs(store_path, exist_ok=True)
       ↓ nio needs the directory to exist before load_store() runs
2. AsyncClient(..., store_path=..., config=ClientConfig(store_sync_tokens=True))
3. client.restore_login(user_id, device_id, access_token)
       ↓ credentials set from env vars — works on first deployment
4. initial_sync = await client.sync(full_state=True)
       ↓ anchors next_batch token; populates rooms.join with prev_batch per room
          and rooms.join[room_id].timeline.events (the current sync window)
5. _backfill(initial_sync)
       ↓ paginates room history backwards from initial_sync prev_batch tokens
       ↓ indexes into Qdrant; collects records from all rooms, sorts by
          timestamp ascending, then appends oldest-first into _buffer so that
          when the deque is full it evicts oldest records, never newest
5.5. _index_initial_sync(initial_sync)
       ↓ indexes initial_sync.rooms.join[*].timeline.events — the slice between
          prev_batch and next_batch that backfill never reaches and sync_forever
          never replays; without this step those messages are silently dropped
6. client.add_event_callback(_on_message, RoomMessageText)
       ↓ registered AFTER backfill + initial-sync indexing to avoid double-indexing
7. asyncio.create_task(client.sync_forever(since=initial_sync.next_batch, ...))
       ↓ live sync starts from the exact token captured before backfill
```

Steps 5, 5.5, and 7 together cover the full timeline without gaps or overlaps: backfill covers everything before `prev_batch`, `_index_initial_sync` covers `prev_batch`→`next_batch`, and `sync_forever` covers everything after `next_batch`. `_seen_event_ids` provides a secondary duplicate guard across all three phases.

---

## SSE fan-out design

`WebhookDispatcher` maintains a `set[asyncio.Queue]`. Each SSE client connection calls `subscribe()` to get its own bounded queue and `unsubscribe()` (in a `finally` block) to remove it on disconnect.

`dispatch()` iterates a **snapshot** of `_subscribers` (`list(self._subscribers)`) so that concurrent subscribe/unsubscribe during iteration is safe. For each subscriber queue that is full, the oldest item is evicted (`get_nowait()`) before the new item is inserted — this gives slow clients a lossy-but-continuous stream instead of stalling the broadcast path.

Queue bound: `SSE_QUEUE_MAXSIZE` (default 100). Eviction is logged at WARNING level.

---

## Qdrant collection schema

Collection name: `QDRANT_COLLECTION` (default `matrix_messages`).

| Field | Value |
|---|---|
| Vector size | 1536 (OpenAI `text-embedding-3-small`) |
| Distance | Cosine |
| Point ID | UUID derived from SHA-256 of `event_id` (first 16 bytes → UUID) — deterministic, so upserting the same event twice is idempotent |

Payload stored per point: `event_id`, `room_id`, `sender` (Matrix ID), `sender_name` (display name), `body`, `timestamp`.

The text passed to the embedding model is `"{sender_name}: {body}"` — sender name is baked into the vector so searches like "what did Alice say about X" surface results by author naturally. `sender` (the raw Matrix ID) is kept separately for filtered queries. Old records indexed before `sender_name` was added fall back to `sender` when read back.

`init_collection()` is idempotent — checks existing collections before creating. Called once at startup before `matrix_client.start()`.

---

## E2EE notes

- Requires `matrix-nio[e2e]` (installs `python-olm`) and the `libolm3` C library at runtime.
- The Olm SQLite database lives at `MATRIX_STORE_PATH`. On a cold start (empty store), the database is created fresh by nio.
- Historical encrypted messages cannot be decrypted on a new deployment — Olm session keys for past messages are not recoverable. New messages in encrypted rooms will work once device trust is established.
- `MATRIX_DEVICE_ID` must not change between restarts. If it does, the Olm store's device identity no longer matches and E2EE will break.
- libolm is not available in the local dev environment (Homebrew Python 3.14). Unit tests run against the plain `matrix-nio` package (no E2EE). The Dockerfile installs `libolm3`/`libolm-dev` for the full stack.

---

## Running tests

```bash
# Create venv and install deps (libolm not required for unit tests)
python3 -m venv .venv
source .venv/bin/activate
pip install matrix-nio mcp qdrant-client openai fastapi "uvicorn[standard]" \
    pydantic-settings httpx anyio sse-starlette \
    pytest pytest-asyncio pytest-mock respx

# Unit tests — no external services
pytest tests/unit/ -v

# Integration tests — Qdrant must be running
docker compose up qdrant -d
pytest tests/integration/ -v
```

`conftest.py` at the project root adds `src/` to `sys.path`, so no `PYTHONPATH` export or editable install is needed to run tests.

Integration tests use a randomly-named Qdrant collection (per test session) and clean it up in a fixture finalizer. They are skipped automatically if Qdrant is not reachable on `QDRANT_HOST:QDRANT_PORT`.

---

## Key configuration defaults

| Setting | Default | Notes |
|---|---|---|
| `BACKFILL_PAGES_MAX` | `10` | `0` = unlimited; each page is `BACKFILL_LIMIT` messages |
| `BACKFILL_LIMIT` | `100` | Messages per `room_messages()` call |
| `MESSAGE_BUFFER_SIZE` | `500` | `deque(maxlen=...)` — oldest entries dropped automatically |
| `SSE_QUEUE_MAXSIZE` | `100` | Per subscriber; drop-oldest on full |
| `MATRIX_STORE_PATH` | `/tmp/nio_store` | Created automatically; use a volume in production |

---

## Useful Matrix / nio references

- [matrix-nio API docs](https://matrix-nio.readthedocs.io/en/latest/nio.html)
- [matrix-nio examples](https://matrix-nio.readthedocs.io/en/latest/examples.html)
- [Matrix client-server spec — /messages pagination](https://spec.matrix.org/latest/client-server-api/#get_matrixclientv3roomsroomidmessages) — explains the `end` token absence as the correct stop condition
- [Matrix client-server spec — /sync](https://spec.matrix.org/latest/client-server-api/#syncing) — explains `next_batch`, `prev_batch`, and `full_state`
