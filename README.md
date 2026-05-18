# nio-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server for [Matrix](https://matrix.org), built on [matrix-nio](https://github.com/poljar/matrix-nio). Gives AI assistants read/write access to Matrix rooms, semantic search over message history, and real-time message notifications.

## Features

- **`get_recent_messages`** — fetch the most recent messages across all joined rooms, with optional filtering by sender or room
- **`search_messages`** — semantic similarity search over all indexed message history via OpenAI embeddings + Qdrant
- **`get_message_context`** — retrieve messages surrounding a specific event (useful after a search hit)
- **`send_message`** — send a text message to any joined room
- **Webhooks** — POST to a configurable URL and/or stream events via SSE whenever a new message arrives
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

The MCP server runs on stdio (connect via an MCP host such as Claude Desktop). The SSE endpoint is available at `http://localhost:8000/events`.

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and fill in the required values.

| Variable | Required | Default | Description |
|---|---|---|---|
| `MATRIX_HOMESERVER_URL` | yes | — | Homeserver URL, e.g. `https://matrix.example.org` |
| `MATRIX_ACCESS_TOKEN` | yes | — | Long-lived access token |
| `MATRIX_USER_ID` | yes | — | Full MXID, e.g. `@bot:example.org` |
| `MATRIX_DEVICE_ID` | yes | — | Device ID — must be stable across restarts for E2EE |
| `MATRIX_STORE_PATH` | no | `/tmp/nio_store` | Path for the Olm E2EE crypto database (created if absent) |
| `QDRANT_HOST` | no | `localhost` | Qdrant hostname (`qdrant` inside Docker Compose) |
| `QDRANT_PORT` | no | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | no | `matrix_messages` | Qdrant collection name |
| `OPENAI_API_KEY` | yes | — | OpenAI API key for embeddings |
| `WEBHOOK_URL` | no | — | URL to POST new-message payloads to |
| `WEBHOOK_SECRET` | no | — | HMAC-SHA256 signing secret; enables `X-Nio-MCP-Signature` header |
| `BACKFILL_LIMIT` | no | `100` | Messages fetched per page per room during startup backfill |
| `BACKFILL_PAGES_MAX` | no | `10` | Maximum backfill pages per room; `0` = full history |
| `MESSAGE_BUFFER_SIZE` | no | `500` | In-memory ring buffer size for `get_recent_messages` |
| `SSE_QUEUE_MAXSIZE` | no | `100` | Per-subscriber SSE event queue cap (oldest dropped when full) |
| `SSE_PORT` | no | `8000` | Port for the SSE and health HTTP endpoints |

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

Returns the `k` most recent messages from the in-memory buffer (populated by backfill and live sync).

```json
{
  "k": 20,
  "sender": "@alice:example.org",
  "room_id": "!abc123:example.org"
}
```

`sender` and `room_id` are optional filters. Returns a list of message objects:

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

Semantic search over all indexed messages. The query is embedded with OpenAI and matched against Qdrant.

```json
{
  "query": "project standup notes",
  "limit": 10
}
```

Returns the same message fields plus a `score` (cosine similarity, 0–1). Use the returned `event_id` and `room_id` with `get_message_context` to retrieve surrounding messages.

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

### `send_message`

Sends a plain-text message to a room.

```json
{
  "room_id": "!abc123:example.org",
  "body": "Hello from the MCP server!"
}
```

## Webhooks

### HTTP POST

When `WEBHOOK_URL` is set, a POST request is sent to that URL for every new message:

```json
{
  "event_id": "$abc:example.org",
  "room_id": "!abc123:example.org",
  "sender": "@alice:example.org",
  "body": "Hello!",
  "timestamp": 1700000000000
}
```

If `WEBHOOK_SECRET` is set, the request includes an `X-Nio-MCP-Signature: sha256=<hmac>` header for verification.

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
┌─────────────────────────────────────────────────────┐
│                   nio-mcp process                   │
│                                                     │
│  ┌─────────────┐    ┌──────────────────────────┐   │
│  │  MCP server │    │  FastAPI (SSE + health)  │   │
│  │   (stdio)   │    │       :8000              │   │
│  └──────┬──────┘    └────────────┬─────────────┘   │
│         │                        │                  │
│         └──────────┬─────────────┘                  │
│                    │                                 │
│         ┌──────────▼──────────┐                     │
│         │   MatrixMCPClient   │                     │
│         │  (nio AsyncClient)  │                     │
│         └──────┬──────────────┘                     │
│                │                                     │
│    ┌───────────┼──────────────┐                     │
│    │           │              │                     │
│  ┌─▼──────┐ ┌─▼────────┐ ┌──▼──────────────┐      │
│  │Qdrant  │ │ OpenAI   │ │WebhookDispatcher│      │
│  │vector  │ │embeddings│ │ HTTP POST + SSE │      │
│  │store   │ │          │ │  per-subscriber │      │
│  └────────┘ └──────────┘ └────────────────┘      │
└─────────────────────────────────────────────────────┘
```

**Startup sequence:**
1. `os.makedirs` ensures the Olm store directory exists
2. `restore_login()` loads credentials from env vars (works on first run — no prior session needed)
3. Initial `sync(full_state=True)` anchors the sync token
4. Backfill: for each joined room, paginate backwards from the initial sync's `prev_batch` token until `end` is absent (Matrix spec) or `BACKFILL_PAGES_MAX` is reached
5. Register live message callback, then `sync_forever(since=<initial token>)` — no gap between backfill and live

**E2EE note:** On a brand-new deployment the Olm store is empty, so messages encrypted before this device joined cannot be decrypted. New messages in encrypted rooms will be decryptable once device trust is established. Plaintext rooms are unaffected.

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

Integration tests require a running Qdrant instance and are skipped automatically if it is not reachable:

```bash
docker compose up qdrant -d
pytest tests/integration/ -v
```

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
└── integration/      # Real Qdrant required
```

### Connecting to Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "matrix": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--env-file", "/path/to/your/.env",
        "nio-mcp"
      ]
    }
  }
}
```

## License

MIT
