"""Tests for retroactive + self-healing sender_name reconciliation.

Covers the SQLite store helpers, the Qdrant payload rewrite, the shared migration
routine, and the live member-event self-heal path.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nio_mcp.migrations import apply_sender_name, reconcile_sender_names
from nio_mcp.models import MessageRecord
from nio_mcp.store import MessageStore
from nio_mcp.vector_store import VectorStore, _sender_search_text


ROOM = "!room:example.org"
GHOST = "@signal_c352f269-1e90-4714-bdbc-0c49cad77e9f:example.org"
LOCALPART = "signal_c352f269-1e90-4714-bdbc-0c49cad77e9f"


def _record(event_id: str, sender: str, sender_name: str) -> MessageRecord:
    return MessageRecord(
        event_id=event_id,
        room_id=ROOM,
        room_name="Trip",
        sender=sender,
        sender_name=sender_name,
        body="booked the hotel",
        timestamp=1785351709449,
    )


@pytest.fixture
def store(tmp_path):
    s = MessageStore(str(tmp_path / "nio_mcp.db"))
    s.open()
    yield s
    s.close()


# --------------------------------------------------------------------------
# MessageStore
# --------------------------------------------------------------------------

def test_update_sender_name_only_touches_stale_rows(store):
    store.insert_message(_record("$a:example.org", GHOST, LOCALPART))
    store.insert_message(_record("$b:example.org", GHOST, LOCALPART))

    changed = store.update_sender_name(ROOM, GHOST, "Alice")
    assert changed == 2

    # Idempotent: a second pass changes nothing.
    assert store.update_sender_name(ROOM, GHOST, "Alice") == 0
    assert all(r.sender_name == "Alice" for r in store.get_recent_messages(10))


def test_get_sender_name_fixes_surfaces_fallback_rows(store):
    store.insert_message(_record("$a:example.org", GHOST, LOCALPART))
    store.upsert_member(ROOM, GHOST, "Alice")
    # A sender already matching its member name must not be surfaced.
    store.insert_message(_record("$b:example.org", "@bob:example.org", "Bob"))
    store.upsert_member(ROOM, "@bob:example.org", "Bob")

    fixes = store.get_sender_name_fixes()
    assert fixes == [(ROOM, GHOST, "Alice")]


def test_get_sender_name_fixes_ignores_senders_without_member_row(store):
    # Departed member (removed from roster) leaves no name to reconcile against.
    store.insert_message(_record("$a:example.org", GHOST, LOCALPART))
    assert store.get_sender_name_fixes() == []


# --------------------------------------------------------------------------
# VectorStore
# --------------------------------------------------------------------------

@pytest.fixture
def vector_store():
    with patch("nio_mcp.vector_store.AsyncQdrantClient") as cls:
        instance = AsyncMock()
        cls.return_value = instance
        vs = VectorStore(host="localhost", port=6333, collection="test_col")
        yield vs, instance


async def test_vector_update_sender_name_sets_payload_by_filter(vector_store):
    vs, client = vector_store
    await vs.update_sender_name(ROOM, GHOST, "Alice")

    client.set_payload.assert_awaited_once()
    kwargs = client.set_payload.await_args.kwargs
    assert kwargs["payload"]["sender_name"] == "Alice"
    assert kwargs["payload"]["sender_search"] == _sender_search_text(GHOST, "Alice")
    assert kwargs["collection_name"] == "test_col"
    # Scoped to this sender in this room via a filter selector.
    assert kwargs["points"] is not None


# --------------------------------------------------------------------------
# migrations
# --------------------------------------------------------------------------

async def test_apply_sender_name_skips_qdrant_when_nothing_changed(store):
    vs = AsyncMock(spec=VectorStore)
    # No messages for this sender -> SQLite reports 0 changed -> no Qdrant write.
    updated = await apply_sender_name(store, vs, ROOM, GHOST, "Alice")
    assert updated == 0
    vs.update_sender_name.assert_not_awaited()


async def test_apply_sender_name_writes_both_stores(store):
    store.insert_message(_record("$a:example.org", GHOST, LOCALPART))
    vs = AsyncMock(spec=VectorStore)

    updated = await apply_sender_name(store, vs, ROOM, GHOST, "Alice")

    assert updated == 1
    vs.update_sender_name.assert_awaited_once_with(ROOM, GHOST, "Alice")


async def test_reconcile_fixes_fallback_and_is_idempotent(store):
    store.insert_message(_record("$a:example.org", GHOST, LOCALPART))
    store.upsert_member(ROOM, GHOST, "Alice")
    vs = AsyncMock(spec=VectorStore)

    fixed = await reconcile_sender_names(store, vs)
    assert fixed == 1
    vs.update_sender_name.assert_awaited_once_with(ROOM, GHOST, "Alice")

    # Second run: names already reconciled, no work, no Qdrant writes.
    vs.update_sender_name.reset_mock()
    assert await reconcile_sender_names(store, vs) == 0
    vs.update_sender_name.assert_not_awaited()


async def test_reconcile_does_not_downgrade_to_fallback_member_name(store):
    # Message already has a real name; member roster only knows the localpart.
    store.insert_message(_record("$a:example.org", GHOST, "Alice"))
    store.upsert_member(ROOM, GHOST, LOCALPART)
    vs = AsyncMock(spec=VectorStore)

    fixed = await reconcile_sender_names(store, vs)
    assert fixed == 0
    vs.update_sender_name.assert_not_awaited()
    assert store.get_recent_messages(10)[0].sender_name == "Alice"
