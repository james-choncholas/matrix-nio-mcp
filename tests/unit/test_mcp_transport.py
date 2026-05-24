"""Transport-level tests for the /mcp HTTP endpoint.

Covers:
- GET/POST/DELETE at /mcp return 200 without any redirect (regression guard
  against the Starlette Mount trailing-slash 307 that existed before this fix)
- /mcp/ redirects back to /mcp (correct direction; clients using the wrong
  path get corrected rather than bounced away)
- RuntimeError when the session manager has not been initialised
- Lifespan wires StreamableHTTPSessionManager with a non-None idle timeout
  (session leak prevention)
"""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

import nio_mcp.server as server_module


class _FakeSessionManager:
    """Records which HTTP methods reached handle_request and always replies 200."""

    def __init__(self):
        self.methods: list[str] = []

    async def handle_request(self, scope, receive, send):
        self.methods.append(scope["method"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


@pytest.fixture()
def mcp_client():
    sm = _FakeSessionManager()
    server_module._session_manager = sm
    client = TestClient(server_module.app, raise_server_exceptions=True)
    yield client, sm
    server_module._session_manager = None


# ---------------------------------------------------------------------------
# No-redirect checks for each MCP method
# ---------------------------------------------------------------------------

def test_get_mcp_is_200_without_redirect(mcp_client):
    client, sm = mcp_client
    r = client.get("/mcp", follow_redirects=False)
    assert r.status_code == 200
    assert sm.methods == ["GET"]


def test_post_mcp_is_200_without_redirect(mcp_client):
    client, sm = mcp_client
    r = client.post("/mcp", follow_redirects=False)
    assert r.status_code == 200
    assert sm.methods == ["POST"]


def test_delete_mcp_is_200_without_redirect(mcp_client):
    client, sm = mcp_client
    r = client.delete("/mcp", follow_redirects=False)
    assert r.status_code == 200
    assert sm.methods == ["DELETE"]


# ---------------------------------------------------------------------------
# Trailing-slash direction check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["get", "post", "delete"])
def test_mcp_trailing_slash_redirects_to_canonical_path(mcp_client, method):
    client, sm = mcp_client
    r = getattr(client, method)("/mcp/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"].rstrip("/") == "http://testserver/mcp"
    assert sm.methods == []  # handle_request must not have been called


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

def test_mcp_endpoint_raises_when_session_manager_none():
    original = server_module._session_manager
    server_module._session_manager = None
    try:
        client = TestClient(server_module.app, raise_server_exceptions=True)
        with pytest.raises(RuntimeError, match="MCP session manager not initialized"):
            client.get("/mcp")
    finally:
        server_module._session_manager = original


# ---------------------------------------------------------------------------
# Lifespan wiring
# ---------------------------------------------------------------------------

def test_lifespan_passes_idle_timeout_to_session_manager():
    """session_idle_timeout must be set so abandoned sessions are reaped."""
    init_kwargs: dict = {}

    @asynccontextmanager
    async def _fake_run():
        yield

    class FakeSM:
        def __init__(self, **kwargs):
            init_kwargs.update(kwargs)
            self.run = _fake_run

    settings = MagicMock()
    settings.mcp_session_timeout = 900

    with (
        patch("nio_mcp.server.StreamableHTTPSessionManager", FakeSM),
        patch("nio_mcp.server.get_settings", return_value=settings),
    ):
        with TestClient(server_module.app):
            pass

    assert init_kwargs["session_idle_timeout"] == 900
    assert init_kwargs["stateless"] is False
