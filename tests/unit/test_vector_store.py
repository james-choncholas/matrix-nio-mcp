import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nio_mcp.vector_store import VectorStore, _event_id_to_uuid
from nio_mcp.models import MessageRecord, SearchResult


RECORD = MessageRecord(
    event_id="$abc123:example.org",
    room_id="!room:example.org",
    sender="@alice:example.org",
    body="Hello world",
    timestamp=1700000000000,
)
VECTOR = [0.1] * 1536


@pytest.fixture
def mock_qdrant():
    with patch("nio_mcp.vector_store.AsyncQdrantClient") as cls:
        instance = AsyncMock()
        cls.return_value = instance
        yield instance


@pytest.fixture
def store(mock_qdrant):
    return VectorStore(host="localhost", port=6333, collection="test_col")


def test_event_id_to_uuid_is_deterministic():
    a = _event_id_to_uuid("$abc:example.org")
    b = _event_id_to_uuid("$abc:example.org")
    assert a == b


def test_event_id_to_uuid_differs_for_different_ids():
    a = _event_id_to_uuid("$aaa:example.org")
    b = _event_id_to_uuid("$bbb:example.org")
    assert a != b


async def test_init_collection_creates_when_absent(store, mock_qdrant):
    existing = MagicMock()
    existing.collections = []
    mock_qdrant.get_collections.return_value = existing
    await store.init_collection()
    mock_qdrant.create_collection.assert_called_once()
    call_kwargs = mock_qdrant.create_collection.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_col"


async def test_init_collection_skips_when_present(store, mock_qdrant):
    col = MagicMock()
    col.name = "test_col"
    existing = MagicMock()
    existing.collections = [col]
    mock_qdrant.get_collections.return_value = existing
    await store.init_collection()
    mock_qdrant.create_collection.assert_not_called()


async def test_upsert_builds_correct_payload(store, mock_qdrant):
    await store.upsert(RECORD, VECTOR)
    mock_qdrant.upsert.assert_called_once()
    call_kwargs = mock_qdrant.upsert.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_col"
    points = call_kwargs["points"]
    assert len(points) == 1
    point = points[0]
    assert point.payload["event_id"] == RECORD.event_id
    assert point.payload["room_id"] == RECORD.room_id
    assert point.payload["sender"] == RECORD.sender
    assert point.payload["body"] == RECORD.body
    assert point.payload["timestamp"] == RECORD.timestamp
    assert point.vector == VECTOR


async def test_upsert_uses_deterministic_id(store, mock_qdrant):
    await store.upsert(RECORD, VECTOR)
    point = mock_qdrant.upsert.call_args.kwargs["points"][0]
    expected_id = _event_id_to_uuid(RECORD.event_id)
    assert str(point.id) == expected_id


async def test_search_maps_hits_to_search_results(store, mock_qdrant):
    hit = MagicMock()
    hit.score = 0.95
    hit.payload = {
        "event_id": "$abc:example.org",
        "room_id": "!room:example.org",
        "sender": "@alice:example.org",
        "body": "Hello",
        "timestamp": 1700000000000,
    }
    mock_qdrant.search.return_value = [hit]
    results = await store.search(VECTOR, limit=5)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].score == 0.95
    assert results[0].body == "Hello"


async def test_search_with_no_filters_passes_no_filter(store, mock_qdrant):
    mock_qdrant.search.return_value = []
    await store.search(VECTOR, limit=10)
    call_kwargs = mock_qdrant.search.call_args.kwargs
    assert call_kwargs["query_filter"] is None


async def test_search_with_room_filter_passes_filter(store, mock_qdrant):
    mock_qdrant.search.return_value = []
    await store.search(VECTOR, room_id="!room:example.org")
    call_kwargs = mock_qdrant.search.call_args.kwargs
    assert call_kwargs["query_filter"] is not None


async def test_close_calls_client_close(store, mock_qdrant):
    await store.close()
    mock_qdrant.close.assert_called_once()
