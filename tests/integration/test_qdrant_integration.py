"""
Integration tests for VectorStore against a real Qdrant instance.

Run with:
    docker compose up qdrant -d
    pytest tests/integration/ -v
"""
import uuid
import pytest
from tests.integration.conftest import skip_if_no_qdrant, QDRANT_HOST, QDRANT_PORT
from nio_mcp.vector_store import VectorStore
from nio_mcp.models import MessageRecord

COLLECTION = f"test_{uuid.uuid4().hex[:8]}"
VECTOR_SIZE = 8  # small for speed in tests


def make_record(i: int, room_id: str = "!room:x", sender: str = "@alice:x") -> MessageRecord:
    return MessageRecord(
        event_id=f"$event{i}:example.org",
        room_id=room_id,
        room_name="Test Room",
        sender=sender,
        sender_name=sender.split(":")[0].lstrip("@"),
        body=f"message number {i}",
        timestamp=1700000000000 + i * 1000,
    )


def make_vector(seed: float, size: int = VECTOR_SIZE) -> list[float]:
    # Simple deterministic unit-ish vector for testing
    return [seed] * size


@pytest.fixture
async def store():
    vs = VectorStore(host=QDRANT_HOST, port=QDRANT_PORT, collection=COLLECTION)
    await vs.init_collection(vector_size=VECTOR_SIZE)
    yield vs
    # Cleanup
    try:
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        await client.delete_collection(COLLECTION)
        await client.close()
    except Exception:
        pass
    await vs.close()


@skip_if_no_qdrant
async def test_init_collection_is_idempotent(store):
    # Calling init_collection again should not raise
    await store.init_collection(vector_size=VECTOR_SIZE)


@skip_if_no_qdrant
async def test_upsert_and_search_returns_record(store):
    record = make_record(1)
    vector = make_vector(0.5)
    await store.upsert(record, vector)

    # Allow Qdrant to index (wait=False)
    import asyncio
    await asyncio.sleep(0.5)

    results = await store.search(vector, limit=5)
    assert len(results) >= 1
    found = next((r for r in results if r.event_id == record.event_id), None)
    assert found is not None
    assert found.body == record.body
    assert found.room_id == record.room_id
    assert found.sender == record.sender


@skip_if_no_qdrant
async def test_upsert_is_idempotent(store):
    record = make_record(2)
    vector = make_vector(0.3)
    await store.upsert(record, vector)
    await store.upsert(record, vector)  # same event_id → same UUID → upsert overwrites

    import asyncio
    await asyncio.sleep(0.5)

    results = await store.search(vector, limit=10)
    matching = [r for r in results if r.event_id == record.event_id]
    assert len(matching) == 1  # no duplicates


@skip_if_no_qdrant
async def test_search_score_ordering(store):
    # Insert two records: one very close to query vector, one far
    close_record = make_record(10)
    far_record = make_record(11)
    query_vector = make_vector(1.0)
    await store.upsert(close_record, make_vector(0.99))
    await store.upsert(far_record, make_vector(0.01))

    import asyncio
    await asyncio.sleep(0.5)

    results = await store.search(query_vector, limit=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score  # highest score first


@skip_if_no_qdrant
async def test_search_with_room_filter(store):
    r1 = make_record(20, room_id="!room_a:x")
    r2 = make_record(21, room_id="!room_b:x")
    vec = make_vector(0.5)
    await store.upsert(r1, vec)
    await store.upsert(r2, vec)

    import asyncio
    await asyncio.sleep(0.5)

    results = await store.search(vec, limit=10, room_id="!room_a:x")
    assert all(r.room_id == "!room_a:x" for r in results)


@skip_if_no_qdrant
async def test_search_with_sender_filter(store):
    r1 = make_record(30, sender="@alice:x")
    r2 = make_record(31, sender="@bob:x")
    vec = make_vector(0.5)
    await store.upsert(r1, vec)
    await store.upsert(r2, vec)

    import asyncio
    await asyncio.sleep(0.5)

    results = await store.search(vec, limit=10, sender="@alice:x")
    assert all(r.sender == "@alice:x" for r in results)


@skip_if_no_qdrant
async def test_search_returns_correct_metadata(store):
    record = make_record(40)
    vector = make_vector(0.7)
    await store.upsert(record, vector)

    import asyncio
    await asyncio.sleep(0.5)

    results = await store.search(vector, limit=1)
    found = next((r for r in results if r.event_id == record.event_id), None)
    assert found is not None
    assert found.timestamp == record.timestamp
    assert isinstance(found.score, float)
    assert 0.0 <= found.score <= 1.0


@skip_if_no_qdrant
async def test_search_with_sender_query_matches_alias_variants(store):
    shared_sender = "@fred:x"
    r1 = MessageRecord(
        event_id="$event_alias_1:example.org",
        room_id="!room:x",
        room_name="Test Room",
        sender=shared_sender,
        sender_name="Fred Flintstone",
        body="meeting notes",
        timestamp=1700000010000,
    )
    r2 = MessageRecord(
        event_id="$event_alias_2:example.org",
        room_id="!room:x",
        room_name="Test Room",
        sender=shared_sender,
        sender_name="fred",
        body="follow up",
        timestamp=1700000011000,
    )
    r3 = MessageRecord(
        event_id="$event_alias_3:example.org",
        room_id="!room:x",
        room_name="Test Room",
        sender="@barney:x",
        sender_name="Barney Rubble",
        body="different sender",
        timestamp=1700000012000,
    )
    vec = make_vector(0.5)
    await store.upsert(r1, vec)
    await store.upsert(r2, vec)
    await store.upsert(r3, vec)

    import asyncio
    await asyncio.sleep(0.5)

    results = await store.search(vec, limit=10, sender_query="Fred Flintstone")
    event_ids = {result.event_id for result in results}
    assert r1.event_id in event_ids
    assert r2.event_id in event_ids
    assert r3.event_id not in event_ids


@skip_if_no_qdrant
async def test_scroll_with_sender_query_returns_matching_sender(store):
    r1 = MessageRecord(
        event_id="$event_scroll_1:example.org",
        room_id="!room:x",
        room_name="Test Room",
        sender="@fred:x",
        sender_name="Fred Flintstone",
        body="latest message",
        timestamp=1700000020000,
    )
    r2 = MessageRecord(
        event_id="$event_scroll_2:example.org",
        room_id="!room:x",
        room_name="Test Room",
        sender="@barney:x",
        sender_name="Barney Rubble",
        body="other sender",
        timestamp=1700000021000,
    )
    vec = make_vector(0.5)
    await store.upsert(r1, vec)
    await store.upsert(r2, vec)

    import asyncio
    await asyncio.sleep(0.5)

    results = await store.scroll(limit=10, sender_query="fred")
    assert len(results) >= 1
    assert all(result.sender == "@fred:x" for result in results)
