import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nio_mcp.vector_store import VectorStore, _event_id_to_uuid, _sender_search_text
from nio_mcp.models import MessageRecord, SearchResult


RECORD = MessageRecord(
    event_id="$abc123:example.org",
    room_id="!room:example.org",
    sender="@alice:example.org",
    sender_name="Alice",
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


def test_sender_search_text_includes_name_sender_and_localpart():
    sender_search = _sender_search_text("@fred.flintstone:example.org", "Fred Flintstone")
    assert "Fred Flintstone" in sender_search
    assert "@fred.flintstone:example.org" in sender_search
    assert "fred.flintstone" in sender_search


async def test_init_collection_creates_when_absent(store, mock_qdrant):
    existing = MagicMock()
    existing.collections = []
    mock_qdrant.get_collections.return_value = existing
    await store.init_collection()
    mock_qdrant.create_collection.assert_called_once()
    assert mock_qdrant.create_payload_index.call_count == 2
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
    assert mock_qdrant.create_payload_index.call_count == 2


async def test_init_collection_creates_timestamp_index(store, mock_qdrant):
    existing = MagicMock()
    existing.collections = []
    mock_qdrant.get_collections.return_value = existing
    await store.init_collection()
    call_kwargs = next(
        call.kwargs
        for call in mock_qdrant.create_payload_index.call_args_list
        if call.kwargs["field_name"] == "timestamp"
    )
    assert call_kwargs["collection_name"] == "test_col"
    schema = call_kwargs["field_schema"]
    assert schema.type == "integer"
    assert schema.is_principal is True


async def test_init_collection_creates_sender_search_index(store, mock_qdrant):
    existing = MagicMock()
    existing.collections = []
    mock_qdrant.get_collections.return_value = existing
    await store.init_collection()
    call_kwargs = next(
        call.kwargs
        for call in mock_qdrant.create_payload_index.call_args_list
        if call.kwargs["field_name"] == "sender_search"
    )
    schema = call_kwargs["field_schema"]
    assert schema.type == "text"
    assert schema.tokenizer == "word"
    assert schema.lowercase is True


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
    assert point.payload["sender_name"] == RECORD.sender_name
    assert point.payload["sender_search"] == "Alice @alice:example.org"
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


async def test_search_with_timestamp_filter_passes_range(store, mock_qdrant):
    mock_qdrant.search.return_value = []
    await store.search(VECTOR, after_ts=1000, before_ts=2000)
    call_kwargs = mock_qdrant.search.call_args.kwargs
    f = call_kwargs["query_filter"]
    assert f is not None
    ts_conditions = [c for c in f.must if c.key == "timestamp"]
    assert len(ts_conditions) == 1
    assert ts_conditions[0].range.gte == 1000
    assert ts_conditions[0].range.lte == 2000


async def test_search_with_sender_query_uses_sender_search_filter(store, mock_qdrant):
    mock_qdrant.search.return_value = []
    await store.search(VECTOR, sender_query="fred")
    call_kwargs = mock_qdrant.search.call_args.kwargs
    f = call_kwargs["query_filter"]
    assert f.min_should is not None
    assert f.min_should.min_count == 1
    assert f.min_should.conditions[0].key == "sender_search"
    assert f.min_should.conditions[0].match.text == "fred"


async def test_search_with_multi_token_sender_query_falls_back_to_broader_match(
    store, mock_qdrant
):
    mock_qdrant.search.side_effect = [[], []]
    await store.search(VECTOR, sender_query="Fred Flintstone")
    assert mock_qdrant.search.call_count == 2
    strict_filter = mock_qdrant.search.call_args_list[0].kwargs["query_filter"]
    fallback_filter = mock_qdrant.search.call_args_list[1].kwargs["query_filter"]
    assert strict_filter.min_should.min_count == 2
    assert fallback_filter.min_should.min_count == 1


async def test_scroll_returns_search_results_with_zero_score(store, mock_qdrant):
    point = MagicMock()
    point.payload = {
        "event_id": "$s:example.org",
        "room_id": "!room:example.org",
        "sender": "@alice:example.org",
        "sender_name": "Alice",
        "body": "scrolled",
        "timestamp": 1700000000000,
    }
    mock_qdrant.scroll.return_value = ([point], None)
    results = await store.scroll(limit=5, after_ts=1000, before_ts=2000)
    assert len(results) == 1
    assert results[0].score == 0.0
    assert results[0].body == "scrolled"


async def test_scroll_orders_by_timestamp_desc(store, mock_qdrant):
    mock_qdrant.scroll.return_value = ([], None)
    await store.scroll(limit=5)
    call_kwargs = mock_qdrant.scroll.call_args.kwargs
    order_by = call_kwargs["order_by"]
    assert order_by.key == "timestamp"
    assert order_by.direction == "desc"


async def test_scroll_with_timestamp_filter_passes_range(store, mock_qdrant):
    mock_qdrant.scroll.return_value = ([], None)
    await store.scroll(after_ts=1000, before_ts=2000)
    call_kwargs = mock_qdrant.scroll.call_args.kwargs
    f = call_kwargs["scroll_filter"]
    assert f is not None
    ts_conditions = [c for c in f.must if c.key == "timestamp"]
    assert ts_conditions[0].range.gte == 1000
    assert ts_conditions[0].range.lte == 2000


async def test_scroll_with_sender_query_uses_sender_search_filter(store, mock_qdrant):
    mock_qdrant.scroll.return_value = ([], None)
    await store.scroll(sender_query="fred")
    call_kwargs = mock_qdrant.scroll.call_args.kwargs
    f = call_kwargs["scroll_filter"]
    assert f.min_should is not None
    assert f.min_should.min_count == 1
    assert f.min_should.conditions[0].key == "sender_search"


async def test_close_calls_client_close(store, mock_qdrant):
    await store.close()
    mock_qdrant.close.assert_called_once()
