import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nio_mcp.models import MessageRecord, SearchResult
import nio_mcp.server as server_module


RECORD = MessageRecord(
    event_id="$evt:example.org",
    room_id="!room:example.org",
    room_name="Test Room",
    sender="@alice:example.org",
    sender_name="Alice",
    body="Hello world",
    timestamp=1700000000000,
)


ROOM_INFO = {
    "room_id": "!room:example.org",
    "name": "Test Room",
    "members": [
        {"mxid": "@alice:example.org", "display_name": "Alice"},
        {"mxid": "@bob:example.org", "display_name": "bob"},
    ],
}


@pytest.fixture(autouse=True)
def mock_matrix_client():
    client = AsyncMock()
    client.get_recent_messages = AsyncMock(return_value=[RECORD])
    client.send_message = AsyncMock(return_value={"event_id": "$sent:example.org"})
    client.get_message_context = AsyncMock(
        return_value={"event": RECORD.to_dict(), "before": [], "after": []}
    )
    client.get_room_info = MagicMock(return_value=ROOM_INFO)
    server_module.app.state.matrix_client = client
    yield client
    server_module.app.state.matrix_client = None


SEARCH_RESULT = SearchResult(
    event_id="$r:x",
    room_id="!r:x",
    sender="@a:x",
    sender_name="A",
    body="Found",
    timestamp=1700000001000,
    score=0.99,
)


@pytest.fixture
def mock_embedding_and_store():
    with (
        patch("nio_mcp.server.EmbeddingClient") as emb_cls,
        patch("nio_mcp.server.VectorStore") as vs_cls,
        patch("nio_mcp.server.get_settings") as gs,
    ):
        settings = MagicMock()
        settings.openai_api_key = "sk-test"
        settings.qdrant_host = "localhost"
        settings.qdrant_port = 6333
        settings.qdrant_collection = "test"
        gs.return_value = settings

        emb_instance = AsyncMock()
        emb_instance.embed = AsyncMock(return_value=[0.1] * 1536)
        emb_cls.return_value = emb_instance

        vs_instance = AsyncMock()
        vs_instance.search = AsyncMock(return_value=[SEARCH_RESULT])
        vs_instance.scroll = AsyncMock(return_value=[SEARCH_RESULT])
        vs_cls.return_value = vs_instance

        yield emb_instance, vs_instance


async def _call(name, arguments):
    results = await server_module.call_tool(name, arguments)
    return json.loads(results[0].text)


async def test_get_recent_messages_delegates_to_client(mock_matrix_client):
    data = await _call("get_recent_messages", {"k": 5})
    mock_matrix_client.get_recent_messages.assert_called_once_with(k=5, sender=None, room_id=None)
    assert isinstance(data, list)
    assert data[0]["event_id"] == RECORD.event_id


async def test_get_recent_messages_passes_filters(mock_matrix_client):
    await _call("get_recent_messages", {"k": 3, "sender": "@bob:x", "room_id": "!r:x"})
    mock_matrix_client.get_recent_messages.assert_called_once_with(
        k=3, sender="@bob:x", room_id="!r:x"
    )


async def test_search_messages_embeds_and_searches(mock_embedding_and_store):
    emb, vs = mock_embedding_and_store
    data = await _call("search_messages", {"query": "project meeting", "limit": 5})
    emb.embed.assert_called_once_with("project meeting")
    _, kwargs = vs.search.call_args
    assert kwargs["limit"] == 5
    assert kwargs["sender_query"] is None
    assert isinstance(data, list)
    assert data[0]["score"] == 0.99


async def test_search_messages_passes_timestamp_filters_to_search(mock_embedding_and_store):
    emb, vs = mock_embedding_and_store
    await _call("search_messages", {"query": "hello", "after_ts": 1000, "before_ts": 2000})
    _, kwargs = vs.search.call_args
    assert kwargs["after_ts"] == 1000
    assert kwargs["before_ts"] == 2000


async def test_search_messages_passes_sender_query_to_search(mock_embedding_and_store):
    emb, vs = mock_embedding_and_store
    await _call("search_messages", {"query": "hello", "sender": "Fred Flintstone"})
    emb.embed.assert_called_once_with("hello")
    _, kwargs = vs.search.call_args
    assert kwargs["sender_query"] == "Fred Flintstone"


async def test_search_messages_time_only_uses_scroll(mock_embedding_and_store):
    emb, vs = mock_embedding_and_store
    data = await _call("search_messages", {"after_ts": 1700000000000, "before_ts": 1700000002000})
    emb.embed.assert_not_called()
    vs.scroll.assert_called_once()
    assert isinstance(data, list)


async def test_search_messages_sender_only_uses_scroll(mock_embedding_and_store):
    emb, vs = mock_embedding_and_store
    data = await _call("search_messages", {"sender": "fred"})
    emb.embed.assert_not_called()
    _, kwargs = vs.scroll.call_args
    assert kwargs["sender_query"] == "fred"
    assert isinstance(data, list)


async def test_search_messages_no_args_returns_error(mock_embedding_and_store):
    data = await _call("search_messages", {})
    assert "error" in data


async def test_search_messages_whitespace_query_uses_scroll(mock_embedding_and_store):
    emb, vs = mock_embedding_and_store
    await _call("search_messages", {"query": "   ", "after_ts": 1700000000000})
    emb.embed.assert_not_called()
    vs.scroll.assert_called_once()


async def test_send_message_delegates_to_client(mock_matrix_client):
    settings = MagicMock()
    settings.allow_send_message = True
    with patch("nio_mcp.server.get_settings", return_value=settings):
        data = await _call("send_message", {"room_id": "!r:x", "body": "Hi!"})
    mock_matrix_client.send_message.assert_called_once_with(room_id="!r:x", body="Hi!")
    assert data["event_id"] == "$sent:example.org"


async def test_send_message_disabled_by_default(mock_matrix_client):
    settings = MagicMock()
    settings.allow_send_message = False
    with patch("nio_mcp.server.get_settings", return_value=settings):
        data = await _call("send_message", {"room_id": "!r:x", "body": "Hi!"})
    mock_matrix_client.send_message.assert_not_called()
    assert "error" in data


async def test_get_message_context_delegates_to_client(mock_matrix_client):
    data = await _call(
        "get_message_context",
        {"room_id": "!r:x", "event_id": "$e:x", "before": 3, "after": 3},
    )
    mock_matrix_client.get_message_context.assert_called_once_with(
        room_id="!r:x", event_id="$e:x", before=3, after=3
    )
    assert "event" in data


async def test_get_room_info_delegates_to_client(mock_matrix_client):
    data = await _call("get_room_info", {"room_id": "!room:example.org"})
    mock_matrix_client.get_room_info.assert_called_once_with(room_id="!room:example.org")
    assert data["room_id"] == "!room:example.org"
    assert data["name"] == "Test Room"
    assert len(data["members"]) == 2


async def test_unknown_tool_returns_error(mock_matrix_client):
    data = await _call("nonexistent_tool", {})
    assert "error" in data


async def test_tool_exception_returns_error_dict(mock_matrix_client):
    mock_matrix_client.get_recent_messages.side_effect = RuntimeError("boom")
    data = await _call("get_recent_messages", {})
    assert "error" in data
    assert "boom" in data["error"]


async def test_call_tool_raises_runtime_error_when_client_not_initialized():
    original = server_module.app.state.matrix_client
    server_module.app.state.matrix_client = None
    try:
        with pytest.raises(RuntimeError, match="Matrix client not initialised"):
            await server_module.call_tool("get_recent_messages", {})
    finally:
        server_module.app.state.matrix_client = original


async def test_sse_endpoint_raises_runtime_error_when_dispatcher_not_initialized():
    original = server_module.app.state.webhook_dispatcher
    server_module.app.state.webhook_dispatcher = None
    try:
        with pytest.raises(RuntimeError, match="Webhook dispatcher not initialised"):
            await server_module.sse_endpoint()
    finally:
        server_module.app.state.webhook_dispatcher = original
