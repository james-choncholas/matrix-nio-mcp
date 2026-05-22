# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Unit tests (no external services needed — uses the pre-built .venv at repo root)
.venv/bin/pytest tests/unit/ -v

# Single test file
.venv/bin/pytest tests/unit/test_vector_store.py -v

# Integration tests (real Qdrant required; auto-skipped if not reachable)
docker compose up qdrant -d
.venv/bin/pytest tests/integration/ -v

# Run the full server stack
docker compose up --build
```

`conftest.py` at the root adds `src/` to `sys.path`, so no editable install or `PYTHONPATH` export is needed when running tests via the `.venv` directly.

## Architecture

`server.py:main()` runs two concurrent tasks via `anyio`: the MCP server (stdio) and a FastAPI/uvicorn server for SSE on `:8000`. All components are constructed in `_run()` and wired into `MatrixMCPClient`:

```
MatrixMCPClient
  ├── EmbeddingClient    — OpenAI text-embedding-3-small; called during backfill and on each live message
  ├── VectorStore        — Qdrant wrapper; upsert on index, search on search_messages tool
  └── WebhookDispatcher  — HTTP POST + per-subscriber asyncio.Queue SSE fan-out
```

`server.py` holds module-level singletons `_matrix_client` and `_webhook_dispatcher`. The `search_messages` tool constructs its own `EmbeddingClient` and `VectorStore` per call (intentional — they are stateless and searches are infrequent).

### Data flow for a new message
`sync_forever` callback → `_on_message` → embed `"{sender_name}: {body}"` → upsert into Qdrant → `WebhookDispatcher.dispatch()`

### Startup paths (in `matrix_client.py:start()`)
There are two distinct paths based on whether a sync token was previously persisted:

- **Restart** (`loaded_sync_token` present): skip backfill, load `buffer.json`, start `sync_forever(since=stored_token)` — Matrix delivers missed events automatically.
- **Fresh start**: `sync(full_state=True)` to anchor position → backfill history via `room_messages()` pagination → `_index_initial_sync` covers the gap between `prev_batch` and `next_batch` → `sync_forever(since=next_batch)`. These three phases cover the full timeline without gaps or overlaps. `_seen_event_ids` is a secondary duplicate guard across all phases.

See `AGENTS.md` for full detail on the startup sequence, matrix-nio API patterns, Qdrant schema, and SSE fan-out design.

### Key data shapes
- `MessageRecord` — all fields stored in Qdrant payload and the in-memory `deque` buffer. `timestamp` is Unix milliseconds.
- `SearchResult` — same fields plus `score` (cosine similarity 0–1).

### Qdrant collection
Point ID is a UUID derived from SHA-256 of `event_id` (first 16 bytes), making upserts idempotent. Payload fields: `event_id`, `room_id`, `sender`, `sender_name`, `body`, `timestamp`. Vector size: 1536.
