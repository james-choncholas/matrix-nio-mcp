"""Transport-level tests for the /mcp HTTP endpoint.

Covers:
- GET/POST/DELETE at /mcp return 200 without any redirect
- /mcp/ redirects back to /mcp
- RuntimeError when the session manager has not been initialised
- Lifespan wires StreamableHTTPSessionManager with a non-None idle timeout
- Lifespan does not block app startup on long Matrix backfills
"""

import pytest
import asyncio
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import MagicMock, patch, AsyncMock
from httpx import AsyncClient, ASGITransport

import nio_mcp.server as server_module


class _FakeSessionManager:
    """Records which HTTP methods reached handle_request and always replies 200."""

    def __init__(self, **kwargs):
        self.methods: list[str] = []

    @asynccontextmanager
    async def run(self):
        yield

    async def handle_request(self, scope, receive, send):
        self.methods.append(scope["method"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


@contextmanager
def _service_patches():
    """Patches the three long-running service constructors so they don't open real connections."""
    fake_vs = MagicMock()
    fake_vs.init_collection = AsyncMock()
    fake_vs.close = AsyncMock()

    fake_wd = MagicMock()
    fake_wd.start = AsyncMock()
    fake_wd.close = AsyncMock()

    fake_mc = MagicMock()
    fake_mc.start = AsyncMock()
    fake_mc.stop = AsyncMock()

    with (
        patch("nio_mcp.server.VectorStore", return_value=fake_vs),
        patch("nio_mcp.server.WebhookDispatcher", return_value=fake_wd),
        patch("nio_mcp.server.MatrixMCPClient", return_value=fake_mc),
        patch("nio_mcp.server.EmbeddingClient", return_value=MagicMock()),
    ):
        yield


@contextmanager
def _startup_patches(fake_sm):
    """Full set of lifespan patches: services + session manager + settings."""
    with _service_patches():
        settings = MagicMock()
        settings.http_auth_token = ""
        settings.mcp_session_timeout = 1800
        settings.embedding_vector_size = 1536
        with (
            patch("nio_mcp.server.StreamableHTTPSessionManager", return_value=fake_sm),
            patch("nio_mcp.server.get_settings", return_value=settings),
        ):
            yield


@pytest.fixture()
async def mcp_client():
    # Force reset of the global session manager to avoid cross-test leaks
    server_module._session_manager = None
    sm = _FakeSessionManager()
    with _startup_patches(sm):
        lifespan_cm = server_module.lifespan(server_module.app)
        await lifespan_cm.__aenter__()
        try:
            async with AsyncClient(
                transport=ASGITransport(app=server_module.app),
                base_url="http://testserver",
            ) as client:
                yield client, sm
        finally:
            await lifespan_cm.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# No-redirect checks for each MCP method
# ---------------------------------------------------------------------------

async def test_get_mcp_is_200_without_redirect(mcp_client):
    client, sm = mcp_client
    r = await client.get("/mcp", follow_redirects=False)
    assert r.status_code == 200
    assert sm.methods == ["GET"]


async def test_post_mcp_is_200_without_redirect(mcp_client):
    client, sm = mcp_client
    r = await client.post("/mcp", follow_redirects=False)
    assert r.status_code == 200
    assert sm.methods == ["POST"]


async def test_delete_mcp_is_200_without_redirect(mcp_client):
    client, sm = mcp_client
    r = await client.delete("/mcp", follow_redirects=False)
    assert r.status_code == 200
    assert sm.methods == ["DELETE"]


# ---------------------------------------------------------------------------
# Trailing-slash direction check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["get", "post", "delete"])
async def test_mcp_trailing_slash_redirects_to_canonical_path(mcp_client, method):
    client, sm = mcp_client
    r = await getattr(client, method)("/mcp/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"].rstrip("/") == "http://testserver/mcp"
    assert sm.methods == []  # handle_request must not have been called


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_endpoint_raises_when_session_manager_none():
    sm = _FakeSessionManager()
    with _startup_patches(sm):
        lifespan_cm = server_module.lifespan(server_module.app)
        await lifespan_cm.__aenter__()
        try:
            async with AsyncClient(
                transport=ASGITransport(app=server_module.app),
                base_url="http://testserver",
            ) as client:
                # Override the session manager after the lifespan has set it.
                server_module._session_manager = None
                with pytest.raises(RuntimeError, match="MCP session manager not initialized"):
                    await client.get("/mcp")
        finally:
            await lifespan_cm.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Lifespan wiring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifespan_passes_idle_timeout_to_session_manager():
    """session_idle_timeout must be set so abandoned sessions are reaped."""
    init_kwargs: dict = {}

    class FakeSM:
        def __init__(self, **kwargs):
            init_kwargs.update(kwargs)

        @asynccontextmanager
        async def run(self):
            yield

    settings = MagicMock()
    settings.mcp_session_timeout = 900
    settings.http_auth_token = ""
    settings.embedding_vector_size = 1536

    with _service_patches():
        with (
            patch("nio_mcp.server.StreamableHTTPSessionManager", FakeSM),
            patch("nio_mcp.server.get_settings", return_value=settings),
        ):
            lifespan_cm = server_module.lifespan(server_module.app)
            await lifespan_cm.__aenter__()
            await lifespan_cm.__aexit__(None, None, None)

    assert init_kwargs["session_idle_timeout"] == 900
    assert init_kwargs["stateless"] is False


@pytest.mark.asyncio
async def test_lifespan_passes_webhook_settings_to_dispatcher():
    fake_vs = MagicMock()
    fake_vs.init_collection = AsyncMock()
    fake_vs.close = AsyncMock()

    fake_wd = MagicMock()
    fake_wd.start = AsyncMock()
    fake_wd.close = AsyncMock()

    fake_mc = MagicMock()
    fake_mc.start = AsyncMock()
    fake_mc.stop = AsyncMock()

    embedding_client = MagicMock()
    sm = _FakeSessionManager()
    settings = MagicMock()
    settings.http_auth_token = ""
    settings.mcp_session_timeout = 1800
    settings.openai_api_key = "sk-test"
    settings.embedding_model = "text-embedding-3-small"
    settings.embedding_vector_size = 1536
    settings.embedding_max_tokens = 8192
    settings.qdrant_host = "qdrant.internal"
    settings.qdrant_port = 6334
    settings.qdrant_collection = "matrix_messages"
    settings.webhook_url = "http://llm.example.com/v1"
    settings.webhook_bearer_token = "secret-token"
    settings.webhook_prompt_header = "Summarize these:"
    settings.webhook_prompt_per_msg = "{sender_name}: {message}"
    settings.webhook_model = "gpt-4.1-mini"
    settings.webhook_cooldown_seconds = 12.5
    settings.sse_queue_maxsize = 42

    webhook_dispatcher_ctor = MagicMock(return_value=fake_wd)

    with (
        patch("nio_mcp.server.VectorStore", return_value=fake_vs),
        patch("nio_mcp.server.WebhookDispatcher", webhook_dispatcher_ctor),
        patch("nio_mcp.server.MatrixMCPClient", return_value=fake_mc),
        patch("nio_mcp.server.EmbeddingClient", return_value=embedding_client),
        patch("nio_mcp.server.StreamableHTTPSessionManager", return_value=sm),
        patch("nio_mcp.server.get_settings", return_value=settings),
    ):
        lifespan_cm = server_module.lifespan(server_module.app)
        await lifespan_cm.__aenter__()
        await lifespan_cm.__aexit__(None, None, None)

    webhook_dispatcher_ctor.assert_called_once_with(
        webhook_url="http://llm.example.com/v1",
        bearer_token="secret-token",
        prompt_header="Summarize these:",
        prompt_per_msg="{sender_name}: {message}",
        model="gpt-4.1-mini",
        cooldown_seconds=12.5,
        queue_maxsize=42,
    )


@pytest.mark.asyncio
async def test_lifespan_does_not_block_on_matrix_startup():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    fake_vs = MagicMock()
    fake_vs.init_collection = AsyncMock()
    fake_vs.close = AsyncMock()

    fake_wd = MagicMock()
    fake_wd.start = AsyncMock()
    fake_wd.close = AsyncMock()

    async def _blocking_start():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    fake_mc = MagicMock()
    fake_mc.start = AsyncMock(side_effect=_blocking_start)
    fake_mc.stop = AsyncMock()

    sm = _FakeSessionManager()
    settings = MagicMock()
    settings.http_auth_token = ""
    settings.mcp_session_timeout = 1800
    settings.embedding_vector_size = 1536

    with (
        patch("nio_mcp.server.VectorStore", return_value=fake_vs),
        patch("nio_mcp.server.WebhookDispatcher", return_value=fake_wd),
        patch("nio_mcp.server.MatrixMCPClient", return_value=fake_mc),
        patch("nio_mcp.server.EmbeddingClient", return_value=MagicMock()),
        patch("nio_mcp.server.StreamableHTTPSessionManager", return_value=sm),
        patch("nio_mcp.server.get_settings", return_value=settings),
    ):
        lifespan_cm = server_module.lifespan(server_module.app)
        await asyncio.wait_for(lifespan_cm.__aenter__(), timeout=0.2)
        try:
            await asyncio.wait_for(started.wait(), timeout=0.2)

            async with AsyncClient(
                transport=ASGITransport(app=server_module.app),
                base_url="http://test",
            ) as ac:
                response = await ac.get("/health")

            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
        finally:
            await asyncio.wait_for(lifespan_cm.__aexit__(None, None, None), timeout=0.2)

    await asyncio.wait_for(cancelled.wait(), timeout=0.2)
    fake_mc.stop.assert_awaited_once()
