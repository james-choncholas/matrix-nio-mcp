import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

import anyio
import uvicorn
from fastapi import FastAPI
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp import types
from sse_starlette.sse import EventSourceResponse

from nio_mcp.config import get_settings
from nio_mcp.embeddings import EmbeddingClient
from nio_mcp.matrix_client import MatrixMCPClient
from nio_mcp.models import MessageRecord
from nio_mcp.vector_store import VectorStore
from nio_mcp.webhook import WebhookDispatcher

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Module-level singletons set during startup
_matrix_client: Optional[MatrixMCPClient] = None
_webhook_dispatcher: Optional[WebhookDispatcher] = None
_session_manager: Optional[StreamableHTTPSessionManager] = None

# --------------------------------------------------------------------------- #
# MCP server                                                                   #
# --------------------------------------------------------------------------- #

mcp = Server("nio-mcp")


@mcp.list_tools()
async def list_tools() -> list[types.Tool]:
    settings = get_settings()
    tools = [
        types.Tool(
            name="get_recent_messages",
            description=(
                "Return the k most recent Matrix messages. "
                "Optionally filter by sender (MXID) and/or room_id."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "k": {"type": "integer", "default": 20, "description": "Number of messages"},
                    "sender": {"type": "string", "description": "Filter by sender MXID"},
                    "room_id": {"type": "string", "description": "Filter by room ID"},
                },
            },
        ),
        types.Tool(
            name="search_messages",
            description=(
                "Search indexed Matrix messages. Provide a natural-language query for semantic "
                "similarity search, a sender name or MXID to narrow results, after_ts/before_ts "
                "(Unix ms) to filter by time, or any combination of those. If no query is "
                "provided, returns up to limit messages in reverse chronological order by "
                "timestamp (no similarity score). At least one of query, sender, after_ts, or "
                "before_ts must be provided."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query"},
                    "sender": {
                        "type": "string",
                        "description": "Sender name or MXID; matched flexibly",
                    },
                    "limit": {"type": "integer", "default": 10, "description": "Max results"},
                    "after_ts": {"type": "integer", "description": "Only return messages after this timestamp (Unix milliseconds)"},
                    "before_ts": {"type": "integer", "description": "Only return messages before this timestamp (Unix milliseconds)"},
                },
            },
        ),
        types.Tool(
            name="get_message_context",
            description=(
                "Fetch messages surrounding a specific event. "
                "Use after search_messages to retrieve context around a found message."
            ),
            inputSchema={
                "type": "object",
                "required": ["room_id", "event_id"],
                "properties": {
                    "room_id": {"type": "string"},
                    "event_id": {"type": "string"},
                    "before": {"type": "integer", "default": 5, "description": "Messages before"},
                    "after": {"type": "integer", "default": 5, "description": "Messages after"},
                },
            },
        ),
    ]
    if settings.allow_send_message:
        tools.append(
            types.Tool(
                name="send_message",
                description="Send a text message to a Matrix room.",
                inputSchema={
                    "type": "object",
                    "required": ["room_id", "body"],
                    "properties": {
                        "room_id": {"type": "string", "description": "Target room ID"},
                        "body": {"type": "string", "description": "Message text"},
                    },
                },
            )
        )
    return tools


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if _matrix_client is None:
        raise RuntimeError("Matrix client not initialised")

    try:
        if name == "get_recent_messages":
            records = await _matrix_client.get_recent_messages(
                k=arguments.get("k", 20),
                sender=arguments.get("sender"),
                room_id=arguments.get("room_id"),
            )
            return [types.TextContent(type="text", text=json.dumps([r.to_dict() for r in records]))]

        if name == "search_messages":
            query = arguments.get("query", "").strip()
            sender = arguments.get("sender", "").strip()
            limit = arguments.get("limit", 10)
            after_ts = arguments.get("after_ts")
            before_ts = arguments.get("before_ts")

            if not query and not sender and after_ts is None and before_ts is None:
                return [types.TextContent(type="text", text=json.dumps({"error": "Provide at least one of: query, sender, after_ts, before_ts"}))]

            settings = get_settings()
            vector_store = VectorStore(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                collection=settings.qdrant_collection,
            )

            if query:
                embedding_client = EmbeddingClient(api_key=settings.openai_api_key)
                vector = await embedding_client.embed(query)
                results = await vector_store.search(
                    vector,
                    limit=limit,
                    sender_query=sender or None,
                    after_ts=after_ts,
                    before_ts=before_ts,
                )
            else:
                results = await vector_store.scroll(
                    limit=limit,
                    sender_query=sender or None,
                    after_ts=after_ts,
                    before_ts=before_ts,
                )

            return [types.TextContent(type="text", text=json.dumps([r.to_dict() for r in results]))]

        if name == "send_message":
            if not get_settings().allow_send_message:
                return [types.TextContent(type="text", text=json.dumps({"error": "send_message is disabled; set ALLOW_SEND_MESSAGE=true to enable"}))]
            result = await _matrix_client.send_message(
                room_id=arguments["room_id"],
                body=arguments["body"],
            )
            return [types.TextContent(type="text", text=json.dumps(result))]

        if name == "get_message_context":
            result = await _matrix_client.get_message_context(
                room_id=arguments["room_id"],
                event_id=arguments["event_id"],
                before=arguments.get("before", 5),
                after=arguments.get("after", 5),
            )
            return [types.TextContent(type="text", text=json.dumps(result))]

        return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as exc:
        logger.exception("Tool %s raised an error", name)
        return [types.TextContent(type="text", text=json.dumps({"error": str(exc)}))]


# --------------------------------------------------------------------------- #
# FastAPI app                                                                  #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session_manager
    settings = get_settings()
    _session_manager = StreamableHTTPSessionManager(
        app=mcp,
        stateless=False,
        session_idle_timeout=settings.mcp_session_timeout,
    )
    async with _session_manager.run():
        yield
    _session_manager = None


app = FastAPI(title="nio-mcp", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/events")
async def sse_endpoint():
    if _webhook_dispatcher is None:
        raise RuntimeError("Webhook dispatcher not initialised")
    q = _webhook_dispatcher.subscribe()

    async def event_generator():
        try:
            while True:
                data = await q.get()
                yield {"data": data}
        finally:
            _webhook_dispatcher.unsubscribe(q)

    return EventSourceResponse(event_generator())


class _MCPASGIApp:
    async def __call__(self, scope, receive, send):
        if _session_manager is None:
            raise RuntimeError("MCP session manager not initialized")
        await _session_manager.handle_request(scope, receive, send)


# add_route with a class-instance endpoint bypasses Starlette's request_response()
# wrapper and the Mount trailing-slash redirect, so /mcp is served directly.
app.add_route("/mcp", _MCPASGIApp(), methods=["GET", "POST", "DELETE"], include_in_schema=False)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

async def _run() -> None:
    global _matrix_client, _webhook_dispatcher

    settings = get_settings()

    embedding_client = EmbeddingClient(api_key=settings.openai_api_key)
    vector_store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=settings.qdrant_collection,
    )
    webhook_dispatcher = WebhookDispatcher(
        webhook_url=settings.webhook_url,
        webhook_secret=settings.webhook_secret,
        queue_maxsize=settings.sse_queue_maxsize,
    )
    matrix_client = MatrixMCPClient(
        config=settings,
        vector_store=vector_store,
        embedding_client=embedding_client,
        webhook_dispatcher=webhook_dispatcher,
    )

    _webhook_dispatcher = webhook_dispatcher
    _matrix_client = matrix_client

    await vector_store.init_collection()
    await matrix_client.start()

    uvicorn_config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=settings.mcp_port,
        log_level="warning",
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    try:
        await uvicorn_server.serve()
    finally:
        await matrix_client.stop()
        await vector_store.close()
        await webhook_dispatcher.close()


def main() -> None:
    anyio.run(_run)


if __name__ == "__main__":
    main()
