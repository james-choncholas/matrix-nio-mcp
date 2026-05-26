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

Entry point: `server.py:main()` starts a uvicorn/FastAPI server. The MCP protocol is served over Streamable HTTP at `/mcp` via `StreamableHTTPSessionManager`. A separate `/events` SSE endpoint streams live Matrix messages for webhook subscribers.

Important server-level behavior: FastAPI startup does **not** wait for `MatrixMCPClient.start()` to finish. The lifespan creates the shared dependencies (`VectorStore`, `WebhookDispatcher`, `MatrixMCPClient`), starts the vector store and webhook dispatcher, then launches `matrix_client.start()` in a background task. This keeps the HTTP server responsive during long backfills. `/health` is therefore a **liveness** endpoint, not a "backfill complete" readiness check: it returns `200 {"status": "ok"}` while bootstrap/backfill is still running, and returns `503` only if Matrix startup failed hard.

---

## Component wiring

`server.py` constructs all components during the FastAPI lifespan. `MatrixMCPClient` and `WebhookDispatcher` are stored on `app.state` for the MCP tool handlers and SSE endpoint respectively; the MCP session manager is kept in the `_session_manager` module global. The dependency graph is:

```
MatrixMCPClient
  ├── EmbeddingClient    (called during backfill and on each live message)
  ├── VectorStore        (upsert during indexing, search during search_messages tool)
  └── WebhookDispatcher  (dispatch() called on each indexed live message)
```

`search_messages` is the one MCP tool that constructs its own `EmbeddingClient` and `VectorStore` per call rather than reusing the shared instances. This is intentional simplicity — searches are infrequent and the clients are stateless. When `query` is absent or whitespace-only, `EmbeddingClient` is not constructed at all and `VectorStore.scroll()` is called instead of `VectorStore.search()`; sender and time filters still apply through Qdrant payload filters.

### Live message indexing flow

For live traffic the path is:

```
sync_forever callback
  → _on_message()
  → append to in-memory buffer
  → persist record to {MATRIX_STORE_PATH}/pending_index.json
  → embed record.body
  → upsert into Qdrant
  → WebhookDispatcher.dispatch()
  → remove record from pending_index.json
```

`pending_index.json` is a small crash-recovery journal for live events. If the process dies after buffering a message but before indexing completes, startup reloads and retries those records idempotently.

### Key data shapes

- `MessageRecord` — canonical message shape used for the in-memory buffer, `pending_index.json`, webhook payloads, and Qdrant payloads. Fields: `event_id`, `room_id`, `sender`, `sender_name`, `body`, `timestamp` (Unix ms).
- `SearchResult` — same fields as `MessageRecord` plus `score` for semantic-search similarity.

---

## matrix-nio API used in this project

### `AsyncClient` construction

```python
from nio import AsyncClient, AsyncClientConfig

client = AsyncClient(
    homeserver="https://matrix.example.org",
    user="@bot:example.org",
    store_path="/data/nio_store",       # directory for Olm SQLite DB
    config=AsyncClientConfig(store_sync_tokens=True),
)
```

`store_path` must be an **existing directory** — nio calls `load_store()` internally which opens a SQLite file there. Create it with `os.makedirs(path, exist_ok=True)` before constructing the client. `AsyncClientConfig(store_sync_tokens=True)` persists the sync token to the store so restarts resume where they left off.

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

Non-obvious nio behavior: `AsyncClient.sync()` falls back to `client.loaded_sync_token` when `since=` is omitted. On the interrupted-backfill retry path, `start()` clears `loaded_sync_token` before calling `sync(full_state=True)` so the request is truly tokenless; otherwise nio would issue an incremental sync from the persisted crash token.

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

`room.room_id` gives the room ID. `event.event_id`, `event.sender`, `event.body`, `event.server_timestamp` are the fields used here. `room.users` is a `dict[str, RoomMember]` keyed by Matrix ID; `RoomMember.display_name` gives the human-readable name used when constructing `sender_name` metadata.

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

Understanding this sequence is critical for any change to `matrix_client.py:start()`. There are three practical startup paths depending on whether nio restored a sync token and whether local bootstrap was previously marked complete.

### Restart path (stored token present, bootstrap complete)

`client.loaded_sync_token` is populated by nio after `restore_login()` when `store_sync_tokens=True` and a prior sync has been persisted to the store. `start()` only takes the fast restart path when that token is non-empty **and** `{MATRIX_STORE_PATH}/backfill_complete` exists:

```
1. os.makedirs + AsyncClient + restore_login  (same as fresh start)
       ↓ loaded_sync_token is now set from the store
2. _is_backfill_complete()
       ↓ checks {MATRIX_STORE_PATH}/backfill_complete
3. _load_buffer()
       ↓ reads {MATRIX_STORE_PATH}/buffer.json (written by stop()) to pre-populate
         _buffer and _seen_event_ids; FileNotFoundError silently ignored (first boot
         after this feature, or file manually deleted)
4. _load_pending_index() + _retry_pending_index()
       ↓ replays any live messages that were journaled to
         {MATRIX_STORE_PATH}/pending_index.json before a prior crash
5. client.add_event_callback(_on_message, RoomMessageText)
6. asyncio.create_task(client.sync_forever(since=loaded_sync_token, ...))
       ↓ Matrix server delivers all events missed during downtime from that token forward
         _seen_event_ids prevents double-insertion if any replayed events were already
         in the loaded buffer or pending-index journal
```

No Matrix backfill is needed — `sync_forever` catches up on missed events, the buffer file provides immediate history for `get_recent_messages()`, and `pending_index.json` replays any live writes that crashed mid-index. If the token is too old and the server has expired it, the sync will fail (not currently handled; treat as a fresh-start edge case).

### Retry path (stored token present, bootstrap incomplete)

This is the failure-recovery path for "first sync succeeded, process died before `_backfill()` / `_index_initial_sync()` finished". nio has already persisted a sync token, but the local sentinel is absent, so `start()` must **not** trust the token as evidence that bootstrap completed:

```
1. os.makedirs + AsyncClient + restore_login
       ↓ loaded_sync_token is restored from the nio store
2. _is_backfill_complete() returns false
       ↓ no {MATRIX_STORE_PATH}/backfill_complete means the previous bootstrap
         never reached the post-indexing commit point
3. client.loaded_sync_token = ""
       ↓ forces the next sync(full_state=True) call to be truly tokenless;
         otherwise nio would silently fall back to the restored token
4. initial_sync = await client.sync(full_state=True)
       ↓ captures a fresh current timeline window and per-room prev_batch tokens
5. _backfill(initial_sync)
6. _index_initial_sync(initial_sync)
7. _mark_backfill_complete()
       ↓ writes {MATRIX_STORE_PATH}/backfill_complete only after both phases finish
8. _load_pending_index() + _retry_pending_index()
       ↓ replays any messages left in {MATRIX_STORE_PATH}/pending_index.json
         from a prior interrupted live-sync phase
9. client.add_event_callback(_on_message, RoomMessageText)
10. asyncio.create_task(client.sync_forever(since=initial_sync.next_batch, ...))
```

This retry path guarantees the process will not permanently skip bootstrap just because nio persisted `next_batch` before local indexing finished. A moved retry anchor may consume part of a bounded backfill budget; that is intentional and controlled by `BACKFILL_PAGES_MAX`.

### Fresh-start path (no stored token)

```
1. os.makedirs(store_path, exist_ok=True)
       ↓ nio needs the directory to exist before load_store() runs
2. AsyncClient(..., store_path=..., config=AsyncClientConfig(store_sync_tokens=True))
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
6. _mark_backfill_complete()
       ↓ writes {MATRIX_STORE_PATH}/backfill_complete only after bootstrap finishes
7. _load_pending_index() + _retry_pending_index()
       ↓ usually a no-op on true first boot, but safe if a prior run crashed after
         live messages had been journaled to {MATRIX_STORE_PATH}/pending_index.json
8. client.add_event_callback(_on_message, RoomMessageText)
       ↓ registered AFTER backfill + initial-sync indexing to avoid double-indexing
9. asyncio.create_task(client.sync_forever(since=initial_sync.next_batch, ...))
       ↓ live sync starts from the exact token captured before backfill
```

Steps 5, 5.5, and 9 together cover the full timeline without gaps or overlaps: backfill covers everything before `prev_batch`, `_index_initial_sync` covers `prev_batch`→`next_batch`, and `sync_forever` covers everything after `next_batch`. `_seen_event_ids` provides a secondary duplicate guard across all three phases. The sentinel write between bootstrap and live sync is the durable "commit point" that distinguishes a complete bootstrap from one that must be retried.

---

## MCP transport

The server uses the MCP Streamable HTTP transport (`StreamableHTTPSessionManager` from `mcp.server.streamable_http_manager`). MCP clients connect via HTTP POST/GET/DELETE to `/mcp`. Sessions are stateful by default — the session manager tracks each client connection and delivers responses over SSE streams within the HTTP session.

The `StreamableHTTPSessionManager` is created in the FastAPI lifespan and stored in the `_session_manager` module global. A thin `_MCPASGIApp` wrapper is mounted at `/mcp` and delegates raw ASGI calls to `session_manager.handle_request()`.

Lifespan ordering matters:
- `vector_store.init_collection()` and `webhook_dispatcher.start()` are awaited before the app begins serving.
- `matrix_client.start()` is launched as a background task and may still be backfilling after the server is already accepting `/mcp`, `/events`, and `/health` requests.
- Shutdown cancels that background startup task if it is still running, then calls `matrix_client.stop()` and closes the other shared services.

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
| Vector size | `EMBEDDING_VECTOR_SIZE` (default 1536, matches `text-embedding-3-small`) |
| Distance | Cosine |
| Point ID | UUID derived from SHA-256 of `event_id` (first 16 bytes → UUID) — deterministic, so upserting the same event twice is idempotent |

Payload stored per point: `event_id`, `room_id`, `sender` (Matrix ID), `sender_name` (display name), `sender_search` (flexible sender lookup text), `body`, `timestamp`.

The text passed to the embedding model is `body` only. `sender_search` stores display name, MXID, and localpart-derived aliases so `search_messages(sender=...)` can do flexible payload-text matching without polluting the semantic vector. `sender` (the raw Matrix ID) is kept separately for exact filters. Old records indexed before `sender_name` was added fall back to `sender` when read back.

`init_collection()` is idempotent — checks existing collections before creating. Called once at startup before `matrix_client.start()`.

### `VectorStore` search vs scroll

`search(vector, ...)` — cosine similarity search via Qdrant's `/search` endpoint. Accepts optional `after_ts` / `before_ts` (Unix ms) which become a `Range` filter on the `timestamp` payload field, combined with any `room_id` / `sender` exact filters and `sender_query` flexible full-text sender matching via `_build_filter()`.

`scroll(...)` — no query vector; uses Qdrant's `/scroll` endpoint. Called by `search_messages` when `query` is absent or whitespace-only. Returns `SearchResult` with `score=0.0`. Accepts the same filter params and orders results by `timestamp` descending.

---

## E2EE notes

- Requires `matrix-nio[e2e]` (installs `python-olm`) and the `libolm3` C library at runtime.
- The Olm SQLite database lives at `MATRIX_STORE_PATH`. On a cold start (empty store), the database is created fresh by nio.
- Historical encrypted messages cannot be decrypted on a new deployment — Olm session keys for past messages are not recoverable. New messages in encrypted rooms will work once device trust is established.
- `MATRIX_DEVICE_ID` must not change between restarts. If it does, the Olm store's device identity no longer matches and E2EE will break.
- libolm is not available in the local dev environment (Homebrew Python 3.14). Unit tests run against the plain `matrix-nio` package (no E2EE). The Dockerfile installs `libolm3`/`libolm-dev` for the full stack.

---

## Running tests

A pre-built `.venv` is checked in at the repo root. Use it directly — no setup needed:

```bash
# Unit tests — no external services
.venv/bin/pytest tests/unit/ -v

# Single unit test file
.venv/bin/pytest tests/unit/test_vector_store.py -v

# Integration tests — always use the script; it starts Synapse + Qdrant first
./scripts/test-matrix-integration.sh

# Even when focusing on one file, still use the script entry point
./scripts/test-matrix-integration.sh tests/integration/test_qdrant_integration.py
```

`conftest.py` at the project root adds `src/` to `sys.path`, so no `PYTHONPATH` export or editable install is needed to run tests.

Do not invoke the integration tests with bare `pytest` unless you have manually reproduced what the script does. `scripts/test-matrix-integration.sh` is the supported entry point: it starts `docker-compose.integration.yml`, waits for Qdrant and Synapse, runs `pytest tests/integration -v "$@"`, and tears the stack down. Integration tests use a randomly-named Qdrant collection (per test session) and clean it up in a fixture finalizer.

### CI and DinD compatibility

The integration tests are designed to run in Docker-in-Docker (DinD) environments (like Gitea Actions / `act`). To ensure compatibility:
- **Avoid bind-mounting local files** (scripts, configs) from the host into containers in `docker-compose.integration.yml`. In DinD, the Docker daemon and the test runner do not share a filesystem, so the daemon cannot find these files.
- **Use custom Dockerfiles** and the `build` directive for services that need local files. This bundles the files into the image via the build context, which is sent over the network to the daemon.
- `scripts/test-matrix-integration.sh` is responsible for calling `docker compose build` before starting the stack.

## Running locally

To run the full local stack from `docker-compose.yml`:

```bash
docker compose up --build
```

## Building and pushing the container manually

```bash
docker build --platform linux/amd64 -t gitea.choncholas.com/cloud/nio-mcp:main .
docker push gitea.choncholas.com/cloud/nio-mcp:main
```

---

## Key configuration defaults

| Setting | Default | Notes |
|---|---|---|
| `BACKFILL_PAGES_MAX` | `10` | `0` = unlimited; each page is `BACKFILL_LIMIT` messages |
| `BACKFILL_LIMIT` | `100` | Messages per `room_messages()` call |
| `MESSAGE_BUFFER_SIZE` | `500` | `deque(maxlen=...)` — oldest entries dropped automatically |
| `SSE_QUEUE_MAXSIZE` | `100` | Per subscriber; drop-oldest on full |
| `MCP_PORT` | `8000` | Port for the HTTP server; MCP at `/mcp`, Matrix event SSE at `/events` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model name |
| `EMBEDDING_VECTOR_SIZE` | `1536` | Must match the chosen model's output dimension; mismatch causes cryptic Qdrant errors |
| `MATRIX_STORE_PATH` | `/tmp/nio_store` | Created automatically; use a volume in production. Also stores `buffer.json` (restart warm-start cache), `pending_index.json` (live-message retry journal), and `backfill_complete` (bootstrap sentinel) |

---

## Useful Matrix / nio references

- [matrix-nio API docs](https://matrix-nio.readthedocs.io/en/latest/nio.html)
- [matrix-nio examples](https://matrix-nio.readthedocs.io/en/latest/examples.html)
- [Matrix client-server spec — /messages pagination](https://spec.matrix.org/latest/client-server-api/#get_matrixclientv3roomsroomidmessages) — explains the `end` token absence as the correct stop condition
- [Matrix client-server spec — /sync](https://spec.matrix.org/latest/client-server-api/#syncing) — explains `next_batch`, `prev_batch`, and `full_state`
