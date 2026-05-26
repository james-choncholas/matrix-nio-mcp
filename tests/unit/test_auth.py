import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock, AsyncMock
from nio_mcp.server import app

@pytest.mark.asyncio
async def test_health_is_public():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_events_requires_auth_when_configured():
    settings = MagicMock()
    settings.http_auth_token = "secret-token"
    settings.sse_queue_maxsize = 100
    
    with patch("nio_mcp.server.get_settings", return_value=settings):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # No auth
            response = await ac.get("/events")
            assert response.status_code == 401
            
            # Wrong auth
            response = await ac.get("/events", headers={"Authorization": "Bearer wrong"})
            assert response.status_code == 401
            
            # Correct auth
            mock_dispatcher = MagicMock()
            app.state.webhook_dispatcher = mock_dispatcher
            
            with patch("nio_mcp.server.EventSourceResponse") as mock_sse:
                from fastapi import Response
                mock_sse.return_value = Response(status_code=200)
                response = await ac.get("/events", headers={"Authorization": "Bearer secret-token"})
                assert response.status_code == 200
            
            app.state.webhook_dispatcher = None

@pytest.mark.asyncio
async def test_mcp_requires_auth_when_configured():
    settings = MagicMock()
    settings.http_auth_token = "secret-token"
    
    async def mock_handle_request(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"OK"})

    with patch("nio_mcp.server.get_settings", return_value=settings):
        # We need to mock _session_manager because it's used in _MCPASGIApp
        with patch("nio_mcp.server._session_manager") as mock_sm:
            mock_sm.handle_request = AsyncMock(side_effect=mock_handle_request)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                # No auth
                response = await ac.post("/mcp")
                assert response.status_code == 401
                assert response.text == "Unauthorized"
                
                # Wrong auth
                response = await ac.post("/mcp", headers={"Authorization": "Bearer wrong"})
                assert response.status_code == 401
                
                # Correct auth
                # handle_request is called, but we just want to see it doesn't return 401
                response = await ac.post("/mcp", headers={"Authorization": "Bearer secret-token"})
                assert response.status_code != 401

@pytest.mark.asyncio
async def test_auth_is_case_insensitive():
    settings = MagicMock()
    settings.http_auth_token = "secret-token"
    
    with patch("nio_mcp.server.get_settings", return_value=settings):
        # 1. Test /events (FastAPI HTTPBearer)
        mock_dispatcher = MagicMock()
        app.state.webhook_dispatcher = mock_dispatcher
        with patch("nio_mcp.server.EventSourceResponse") as mock_sse:
            from fastapi import Response
            mock_sse.return_value = Response(status_code=200)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                for scheme in ["Bearer", "bearer", "BEARER"]:
                    response = await ac.get("/events", headers={"Authorization": f"{scheme} secret-token"})
                    assert response.status_code == 200, f"Failed on /events with {scheme}"
        app.state.webhook_dispatcher = None

        # 2. Test /mcp (Custom ASGI check)
        async def mock_handle_request(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"OK"})

        with patch("nio_mcp.server._session_manager") as mock_sm:
            mock_sm.handle_request = AsyncMock(side_effect=mock_handle_request)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                for scheme in ["Bearer", "bearer", "BEARER"]:
                    response = await ac.post("/mcp", headers={"Authorization": f"{scheme} secret-token"})
                    assert response.status_code == 200, f"Failed on /mcp with {scheme}"


@pytest.mark.asyncio
async def test_auth_not_required_when_not_configured():
    settings = MagicMock()
    settings.http_auth_token = ""
    
    async def mock_handle_request(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"OK"})

    with patch("nio_mcp.server.get_settings", return_value=settings):
        with patch("nio_mcp.server._session_manager") as mock_sm:
            mock_sm.handle_request = AsyncMock(side_effect=mock_handle_request)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/mcp")
                assert response.status_code != 401
