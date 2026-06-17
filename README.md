# nio-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server for [Matrix](https://matrix.org), built on [matrix-nio](https://github.com/poljar/matrix-nio). Gives AI assistants read/write access to Matrix rooms, semantic search over message history, and real-time message notifications.

## Features

- **`get_recent_messages`** — fetch the most recent messages across all joined rooms, with optional filtering by exact MXID sender or room
- **`search_messages`** — search indexed message history by semantic similarity (OpenAI embeddings + cosine similarity), fuzzy sender name, time range, or any combination
- **`get_message_context`** — retrieve messages surrounding a specific event (useful after a search hit)
- **`get_room_info`** — return the friendly display name and full member list (MXID + display name) for a room
- **`send_message`** — send a text message to any joined room
- **LLM callback** — call any OpenAI-compatible endpoint with a configurable prompt after a cooldown period; multiple messages are batched; also stream events via SSE
- **E2EE support** — works with encrypted rooms via libolm
- **Backfill on startup** — indexes historical messages from all joined rooms before going live

## Requirements

- A Matrix account with a long-lived access token and a stable device ID
- An OpenAI API key (for `text-embedding-3-small` embeddings)
- Docker and Docker Compose (for running the server and Qdrant)

## Quick start

```bash
git clone <this repo>
cd nio-mcp
cp .env.example .env
# Edit .env — fill in MATRIX_*, OPENAI_API_KEY at minimum
docker compose up --build
```

The MCP server is available at `http://localhost:8000/mcp` (Streamable HTTP transport). The Matrix event SSE stream is at `http://localhost:8000/events`.

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and fill in the required values.

| Variable | Required | Default | Description |
|---|---|---|---|
| `MATRIX_HOMESERVER_URL` | yes | — | Homeserver URL, e.g. `https://matrix.example.org` |
| `MATRIX_ACCESS_TOKEN` | yes | — | Long-lived access token |
| `MATRIX_USER_ID` | yes | — | Full MXID, e.g. `@bot:example.org` |
| `MATRIX_DEVICE_ID` | yes | — | Device ID — must be stable across restarts for E2EE |
| `MATRIX_STORE_PATH` | no | `~/.cache/nio-mcp/store` | Path for the Olm E2EE crypto database (created if absent) |
| `MATRIX_KEY_BACKUP_FILE` | no | — | Path to an Element-exported E2EE key file; see [Decrypting historical messages](#decrypting-historical-messages-in-encrypted-rooms) |
| `MATRIX_KEY_BACKUP_PASSPHRASE` | no | — | Passphrase chosen when exporting; required when `MATRIX_KEY_BACKUP_FILE` is set |
| `QDRANT_HOST` | no | `localhost` | Qdrant hostname (`qdrant` inside Docker Compose) |
| `QDRANT_PORT` | no | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | no | `matrix_messages` | Qdrant collection name |
| `OPENAI_API_KEY` | yes | — | OpenAI API key for embeddings |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` | OpenAI embedding model; `dimensions` is only supported by `text-embedding-3-*` models |
| `EMBEDDING_VECTOR_SIZE` | no | `1536` | Output dimension requested from the model and used for the Qdrant collection; see note below |
| `WEBHOOK_URL` | no | — | OpenAI-compatible base URL for the LLM callback (e.g. `https://api.openai.com/v1`) |
| `WEBHOOK_BEARER_TOKEN` | no | — | Bearer token sent in the `Authorization` header |
| `WEBHOOK_PROMPT_HEADER` | no | `New Matrix messages:` | Text prepended once before all per-message lines |
| `WEBHOOK_PROMPT_PER_MSG` | no | `{sender_name} ({sender}) in {room_name} ({room}): {message}` | Template rendered once per buffered message |
| `WEBHOOK_MODEL` | no | `gpt-4o-mini` | Model name passed to the LLM |
| `WEBHOOK_COOLDOWN_SECONDS` | no | `300` | Seconds of silence before the LLM is called; multiple messages within the window are batched |
| `WEBHOOK_TOOLS` | no | — | Optional JSON string of tools/parameters merged into the chat completions request body |
| `BACKFILL_LIMIT` | no | `100` | Messages fetched per page per room during startup backfill |
| `BACKFILL_PAGES_MAX` | no | `10` | Maximum backfill pages per room; `0` = full history |
| `MESSAGE_BUFFER_SIZE` | no | `500` | In-memory ring buffer size for `get_recent_messages` |
| `MATRIX_SYNC_TIMEOUT_MS` | no | `30000` | Matrix `/sync` long-poll timeout in milliseconds |
| `SSE_QUEUE_MAXSIZE` | no | `100` | Per-subscriber SSE event queue cap (oldest dropped when full) |
| `MCP_PORT` | no | `8000` | Port for the HTTP server; MCP at `/mcp`, Matrix event SSE at `/events`, health at `/health` |
| `HTTP_AUTH_TOKEN` | no | — | If set, requires `Authorization: Bearer <token>` for all endpoints |
| `ALLOW_SEND_MESSAGE` | no | `false` | Set to `true` to enable the `send_message` tool |
| `IGNORED_ROOMS` | no | — | Comma-separated list of Matrix room IDs to exclude from indexing, backfill, and live sync (e.g. `!abc:example.org,!def:example.org`); leave empty to include all rooms |

> **Changing `EMBEDDING_MODEL` or `EMBEDDING_VECTOR_SIZE`** requires wiping the Qdrant collection and re-syncing from scratch. The collection is created at startup with the configured vector size; vectors already stored at a different dimension will cause Qdrant errors that cannot be recovered without dropping the collection. To reset: stop the server, delete the Qdrant collection (or point `QDRANT_COLLECTION` at a new name), delete `MATRIX_STORE_PATH/backfill_complete`, then restart.

### Obtaining credentials

**Access token and device ID** — the easiest way is via Element:
1. Log in to Element as your bot account
2. Go to **Settings → Security & Privacy → Session Manager**
3. Copy the access token and session/device ID for your current session

Alternatively, call the Matrix login endpoint directly:

```bash
curl -XPOST 'https://matrix.example.org/_matrix/client/v3/login' \
  -H 'Content-Type: application/json' \
  -d '{"type":"m.login.password","user":"@bot:example.org","password":"secret"}'
```

The response contains `access_token` and `device_id`.

## MCP tools

### `get_recent_messages`

Returns the `k` most recent messages from the in-memory buffer (populated by backfill and live sync). During the initial startup backfill, this endpoint returns an empty array until the backfill phase has finished populating the buffer.

```json
{
  "k": 20,
  "sender": "@alice:example.org",
  "room_id": "!abc123:example.org"
}
```

`sender` and `room_id` are optional filters. `sender` must be an exact MXID (e.g. `@alice:example.org`) — partial names are not matched. Returns a list of message objects:

```json
[
  {
    "event_id": "$abc:example.org",
    "room_id": "!abc123:example.org",
    "sender": "@alice:example.org",
    "body": "Hello!",
    "timestamp": 1700000000000
  }
]
```

### `search_messages`

Search indexed messages by semantic similarity, sender, time range, or any combination of those. At least one of `query`, `sender`, `after_ts`, or `before_ts` must be provided.

```json
{
  "query": "project standup notes",
  "sender": "fred",
  "limit": 10,
  "after_ts": 1700000000000,
  "before_ts": 1700086400000
}
```

- **`query`** — natural-language search; embedded with OpenAI and matched by cosine similarity against Qdrant.
- **`sender`** — fuzzy sender filter. A full MXID (`@alice:example.org`) is matched exactly; anything else (e.g. `alice`, `fred`) uses word search against the sender's MXID, display name, and localpart variants.
- **`after_ts` / `before_ts`** — Unix millisecond timestamps (optional). Filter results to a time window.
- If `query` is omitted, up to `limit` matching messages are returned newest-first by timestamp with `score: 0`.

Returns the same message fields as `get_recent_messages` plus a `score` (cosine similarity, 0–1, or 0 for time-only queries). Use the returned `event_id` and `room_id` with `get_message_context` to retrieve surrounding messages.

### `get_message_context`

Fetches messages before and after a specific event via the Matrix `/context` endpoint.

```json
{
  "room_id": "!abc123:example.org",
  "event_id": "$found_event:example.org",
  "before": 5,
  "after": 5
}
```

### `get_room_info`

Returns the friendly display name and full member list for a room, read from nio's in-memory room state populated during initial sync.

```json
{
  "room_id": "!abc123:example.org"
}
```

Returns:

```json
{
  "room_id": "!abc123:example.org",
  "name": "My Room",
  "members": [
    {"user_id": "@alice:example.org", "display_name": "Alice"},
    {"user_id": "@bob:example.org", "display_name": "Bob"}
  ]
}
```

### `send_message`

Sends a plain-text message to a room.

```json
{
  "room_id": "!abc123:example.org",
  "body": "Hello from the MCP server!"
}
```

## Webhooks

### LLM callback

When `WEBHOOK_URL` is set, an OpenAI-compatible chat-completions request is sent after a configurable cooldown period with no new messages (default 5 minutes). Multiple messages arriving within the cooldown window are batched into a single call.

```
POST {WEBHOOK_URL}/chat/completions
Authorization: Bearer {WEBHOOK_BEARER_TOKEN}
Content-Type: application/json

{
  "model": "gpt-4o-mini",
  "messages": [{ "role": "user", "content": "<rendered prompt>" }]
}
```

**`WEBHOOK_PROMPT_HEADER`** is prepended once. **`WEBHOOK_PROMPT_PER_MSG`** is rendered for every buffered message and the results are joined with newlines. Braces inside message bodies are never re-interpreted as placeholders.

| Placeholder | Value |
|---|---|
| `{message}` | Body of the message |
| `{sender_name}` | Display name of the sender |
| `{sender}` | MXID of the sender |
| `{room_name}` | Display name of the room |
| `{room}` | Room MXID |

Example:
```
WEBHOOK_PROMPT_HEADER=Summarize these Matrix messages and list action items:
WEBHOOK_PROMPT_PER_MSG={sender_name} said: {message}
```
Produces a user message like:
```
Summarize these Matrix messages and list action items:
Alice said: Can we move the standup?
Bob said: Sure, how about 10am?
```

### SSE stream

Connect to `http://localhost:8000/events` to receive a live stream of new messages:

```bash
curl -N http://localhost:8000/events
```

Each event is a JSON-encoded message object. Multiple clients can connect simultaneously — each gets its own independent stream. If a client falls behind by more than `SSE_QUEUE_MAXSIZE` events, the oldest queued events are dropped (the stream remains live but is lossy under load).

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      nio-mcp process                     │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │         FastAPI  :8000                             │  │
│  │   /mcp  (MCP Streamable HTTP)                      │  │
│  │   /events  (Matrix message SSE fan-out)            │  │
│  │   /health                                          │  │
│  └───────────────────────┬────────────────────────────┘  │
│                          │                               │
│             ┌────────────▼────────────┐                  │
│             │     MatrixMCPClient     │                  │
│             │    (nio AsyncClient)    │                  │
│             └──────┬─────────────────┘                  │
│                    │                                     │
│       ┌────────────┼──────────────┐                     │
│       │            │              │                     │
│  ┌────▼───┐  ┌─────▼────┐  ┌─────▼───────────┐        │
│  │Qdrant  │  │ OpenAI   │  │WebhookDispatcher│        │
│  │vector  │  │embeddings│  │ LLM call + SSE  │        │
│  │store   │  │          │  │  per-subscriber │        │
│  └────────┘  └──────────┘  └────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

**Startup sequence:**
1. `os.makedirs` ensures the Olm store directory exists
2. `restore_login()` loads credentials from env vars (works on first run — no prior session needed)
3. Initial `sync(full_state=True)` anchors the sync token
4. Backfill: for each joined room, paginate backwards from the initial sync's `prev_batch` token until `end` is absent (Matrix spec) or `BACKFILL_PAGES_MAX` is reached
5. Register live message callback, then `sync_forever(since=<initial token>)` — no gap between backfill and live

**E2EE note:** On a brand-new deployment the Olm store is empty, so messages encrypted before this device joined cannot be decrypted. New messages in encrypted rooms will be decryptable once device trust is established. Plaintext rooms are unaffected. To recover historical encrypted messages see [Decrypting historical messages](#decrypting-historical-messages-in-encrypted-rooms).

## Decrypting historical messages in encrypted rooms

Matrix Megolm session keys are distributed once at send-time to all devices present in the room. Because the bot's device wasn't present when those sessions were created, it cannot decrypt historical ciphertext — the homeserver only stores encrypted blobs.

Element (and other standard clients) let you export all session keys you hold to an encrypted file. Importing that file into nio-mcp gives the bot the keys it needs to decrypt backfilled history.

### One-time setup

1. In Element, go to **Settings → Security & Privacy → Export E2E room keys**.
2. Choose a passphrase and save the exported `.txt` file.
3. Mount the file into the container and set the two config variables:

   ```yaml
   services:
     nio-mcp:
       volumes:
         - ./element_keys.txt:/data/element_keys.txt:ro
       environment:
         MATRIX_KEY_BACKUP_FILE: /data/element_keys.txt
         MATRIX_KEY_BACKUP_PASSPHRASE: your-export-passphrase
   ```

4. Start (or restart) the bot. On the first run the keys are imported into the Olm store and a sentinel file (`key_backup_imported`) is written to `MATRIX_STORE_PATH`. Subsequent restarts skip the import automatically — the env vars and volume mount can be left in place or removed; either way the import will not run again.

### Notes

- Both `MATRIX_KEY_BACKUP_FILE` and `MATRIX_KEY_BACKUP_PASSPHRASE` must be set together.
- The import happens before backfill, so session keys are available when historical messages are paginated.
- If the file path or passphrase is wrong, startup fails with a clear error rather than silently skipping messages.
- To force a re-import (e.g. after exporting a newer key file), delete `MATRIX_STORE_PATH/key_backup_imported` and restart.

## Development

### Running tests

Unit tests have no external dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install matrix-nio mcp qdrant-client openai fastapi "uvicorn[standard]" \
    pydantic-settings httpx anyio sse-starlette \
    pytest pytest-asyncio pytest-mock respx

pytest tests/unit/ -v
```

Integration tests must be run through the helper script, because it brings up the Docker-backed Matrix homeserver and Qdrant before invoking pytest:

```bash
scripts/test-matrix-integration.sh

# Still use the script entry point even when focusing on one integration file
scripts/test-matrix-integration.sh tests/integration/test_qdrant_integration.py
```

Do not invoke the integration tests with bare `pytest` unless you have already recreated that environment yourself. The script is the supported entry point; it starts `docker-compose.integration.yml`, waits for Synapse and Qdrant, runs `pytest tests/integration -v "$@"`, and tears the stack down again.

### Project layout

```
src/nio_mcp/
├── config.py         # Pydantic Settings
├── models.py         # MessageRecord, SearchResult
├── embeddings.py     # OpenAI embedding client
├── vector_store.py   # Qdrant wrapper
├── matrix_client.py  # nio AsyncClient wrapper
├── webhook.py        # HTTP POST + SSE dispatcher
└── server.py         # MCP server + FastAPI app

tests/
├── unit/             # All external I/O mocked
└── integration/      # Real Qdrant + optional real Matrix homeserver coverage
```

### Connecting to Claude Desktop

Start the server with `docker compose up --build`, then point Claude Desktop at the HTTP endpoint by adding to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "matrix": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## License

MIT
