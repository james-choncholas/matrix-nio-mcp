import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nio_mcp.models import MessageRecord, SearchResult
import nio_mcp.server as server_module


RECORD = MessageRecord(
    event_id="$evt:example.org",
    room_id="!room:example.org",
    sender="@alice:example.org",
    sender_name="Alice",
    body="Hello world",
    timestamp=1700000000000,
)


@pytest.fixture(autouse=True)
def mock_matrix_client():
    client = AsyncMock()
    client.get_recent_messages = AsyncMock(return_value=[RECORD])
    client.send_message = AsyncMock(return_value={"event_id": "$sent:example.org"})
    client.get_message_context = AsyncMock(
        return_value={"event": RECORD.to_dict(), "before": [], "after": []}
    )
    server_module._matrix_client = client
    yield client
    server_module._matrix_client = None


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
        vs_instance.search = AsyncMock(
            return_value=[
                SearchResult(
                    event_id="$r:x",
                    room_id="!r:x",
                    sender="@a:x",
                    sender_name="A",
                    body="Found",
                    timestamp=1,
                    score=0.99,
                )
            ]
        )
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
    vs.search.assert_called_once()
    assert isinstance(data, list)
    assert data[0]["score"] == 0.99


async def test_send_message_delegates_to_client(mock_matrix_client):
    data = await _call("send_message", {"room_id": "!r:x", "body": "Hi!"})
    mock_matrix_client.send_message.assert_called_once_with(room_id="!r:x", body="Hi!")
    assert data["event_id"] == "$sent:example.org"


async def test_get_message_context_delegates_to_client(mock_matrix_client):
    data = await _call(
        "get_message_context",
        {"room_id": "!r:x", "event_id": "$e:x", "before": 3, "after": 3},
    )
    mock_matrix_client.get_message_context.assert_called_once_with(
        room_id="!r:x", event_id="$e:x", before=3, after=3
    )
    assert "event" in data


async def test_unknown_tool_returns_error(mock_matrix_client):
    data = await _call("nonexistent_tool", {})
    assert "error" in data


async def test_tool_exception_returns_error_dict(mock_matrix_client):
    mock_matrix_client.get_recent_messages.side_effect = RuntimeError("boom")
    data = await _call("get_recent_messages", {})
    assert "error" in data
    assert "boom" in data["error"]
