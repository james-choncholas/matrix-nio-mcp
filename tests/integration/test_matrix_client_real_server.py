import asyncio
import json
import os

import pytest

from tests.integration.conftest import (
    FakeEmbeddingClient,
    skip_if_no_matrix,
    skip_if_no_qdrant,
    wait_until,
)


pytestmark = [
    pytest.mark.integration,
    skip_if_no_qdrant,
    skip_if_no_matrix,
]


async def _room_records(vector_store, room_id: str, limit: int = 100):
    return await vector_store.scroll(limit=limit, room_id=room_id)


async def _event_ids_in_store(vector_store, room_id: str, expected_ids: set[str], limit: int = 100):
    records = await _room_records(vector_store, room_id, limit=limit)
    actual_ids = {record.event_id for record in records}
    if expected_ids.issubset(actual_ids):
        return records
    return None


async def _stop_client(client, dispatcher) -> None:
    try:
        await client.stop()
    finally:
        await dispatcher.close()


async def test_start_backfills_history_and_preserves_sender_names(
    matrix_api,
    make_matrix_config,
    make_vector_store,
    make_matrix_client,
):
    alice = await matrix_api.register_user("alice", display_name="Alice")
    bot = await matrix_api.register_user("bot", display_name="Bot")
    room_id = await matrix_api.create_room(alice, name="History Room")
    await matrix_api.invite_and_join(room_id, alice, bot)

    expected_ids = set()
    for index in range(8):
        event_id = await matrix_api.send_text(room_id, alice, f"history message {index}")
        expected_ids.add(event_id)

    vector_store = await make_vector_store()
    config = make_matrix_config(bot, backfill_limit=2, backfill_pages_max=0)
    client, _embedder, dispatcher = make_matrix_client(config, vector_store)

    try:
        await client.start()

        async def backfilled_buffer():
            recent = await client.get_recent_messages(k=20, room_id=room_id)
            if expected_ids.issubset({record.event_id for record in recent}):
                return recent
            return None

        recent = await wait_until(
            backfilled_buffer,
            description="buffer to contain backfilled room history",
        )
        assert {record.sender_name for record in recent if record.sender == alice.user_id} == {"Alice"}

        stored = await wait_until(
            lambda: _event_ids_in_store(vector_store, room_id, expected_ids),
            description="Qdrant to contain backfilled room history",
        )
        assert {record.sender_name for record in stored if record.sender == alice.user_id} == {"Alice"}
    finally:
        await _stop_client(client, dispatcher)


async def test_live_message_is_buffered_indexed_and_dispatched(
    matrix_api,
    make_matrix_config,
    make_vector_store,
    make_matrix_client,
):
    alice = await matrix_api.register_user("alice", display_name="Alice")
    bot = await matrix_api.register_user("bot", display_name="Bot")
    room_id = await matrix_api.create_room(alice, name="Live Room")
    await matrix_api.invite_and_join(room_id, alice, bot)

    vector_store = await make_vector_store()
    config = make_matrix_config(bot)
    client, _embedder, dispatcher = make_matrix_client(config, vector_store)
    queue = dispatcher.subscribe()

    try:
        await client.start()
        event_id = await matrix_api.send_text(room_id, alice, "live message from Alice")

        async def buffered_record():
            recent = await client.get_recent_messages(k=10, room_id=room_id)
            for record in recent:
                if record.event_id == event_id:
                    return record
            return None

        record = await wait_until(buffered_record, description="live message in buffer")
        assert record.sender_name == "Alice"

        payload = await asyncio.wait_for(queue.get(), timeout=5.0)
        data = json.loads(payload)
        assert data["event_id"] == event_id
        assert data["sender_name"] == "Alice"

        stored = await wait_until(
            lambda: _event_ids_in_store(vector_store, room_id, {event_id}),
            description="Qdrant to contain live message",
        )
        live_record = next(record for record in stored if record.event_id == event_id)
        assert live_record.body == "live message from Alice"
    finally:
        dispatcher.unsubscribe(queue)
        await _stop_client(client, dispatcher)


async def test_restart_resumes_from_stored_token_without_buffer_duplicates(
    matrix_api,
    make_matrix_config,
    make_vector_store,
    make_matrix_client,
):
    alice = await matrix_api.register_user("alice", display_name="Alice")
    bot = await matrix_api.register_user("bot", display_name="Bot")
    room_id = await matrix_api.create_room(alice, name="Restart Room")
    await matrix_api.invite_and_join(room_id, alice, bot)

    vector_store = await make_vector_store()
    config = make_matrix_config(bot, store_name="restart_store")

    client_one, _embedder_one, dispatcher_one = make_matrix_client(config, vector_store)
    offline_event_ids = set()

    try:
        await client_one.start()
        first_event_id = await matrix_api.send_text(room_id, alice, "message before restart")
        await wait_until(
            lambda: _event_ids_in_store(vector_store, room_id, {first_event_id}),
            description="initial live message before restart",
        )
    finally:
        await _stop_client(client_one, dispatcher_one)

    assert os.path.exists(os.path.join(config.matrix_store_path, "backfill_complete"))
    assert os.path.exists(os.path.join(config.matrix_store_path, "buffer.json"))

    for index in range(2):
        event_id = await matrix_api.send_text(room_id, alice, f"offline message {index}")
        offline_event_ids.add(event_id)

    client_two, _embedder_two, dispatcher_two = make_matrix_client(config, vector_store)
    try:
        await client_two.start()

        async def recent_with_offline_messages():
            recent = await client_two.get_recent_messages(k=20, room_id=room_id)
            if offline_event_ids.issubset({record.event_id for record in recent}):
                return recent
            return None

        recent = await wait_until(
            recent_with_offline_messages,
            description="recent messages after restart",
        )
        recent_ids = [record.event_id for record in recent]
        assert len(recent_ids) == len(set(recent_ids))

        stored = await wait_until(
            lambda: _event_ids_in_store(vector_store, room_id, offline_event_ids),
            description="Qdrant to contain offline messages after restart",
        )
        stored_ids = [record.event_id for record in stored]
        assert len(stored_ids) == len(set(stored_ids))
    finally:
        await _stop_client(client_two, dispatcher_two)


async def test_pending_index_replays_after_restart(
    matrix_api,
    make_matrix_config,
    make_vector_store,
    make_matrix_client,
):
    alice = await matrix_api.register_user("alice", display_name="Alice")
    bot = await matrix_api.register_user("bot", display_name="Bot")
    room_id = await matrix_api.create_room(alice, name="Pending Index Room")
    await matrix_api.invite_and_join(room_id, alice, bot)

    vector_store = await make_vector_store()
    config = make_matrix_config(bot, store_name="pending_index_store")
    failing_text = "Alice: pending retry message"

    client_one, _embedder_one, dispatcher_one = make_matrix_client(
        config,
        vector_store,
        embedding_client=FakeEmbeddingClient(fail_once_texts={failing_text}),
    )

    try:
        await client_one.start()
        event_id = await matrix_api.send_text(room_id, alice, "pending retry message")

        async def pending_journal_contains_event():
            if event_id in client_one._pending_index:
                return True
            return False

        await wait_until(
            pending_journal_contains_event,
            description="pending index journal to contain failed live message",
        )
        stored = await _room_records(vector_store, room_id)
        assert event_id not in {record.event_id for record in stored}
    finally:
        await _stop_client(client_one, dispatcher_one)

    pending_path = os.path.join(config.matrix_store_path, "pending_index.json")
    with open(pending_path) as pending_file:
        pending_data = json.load(pending_file)
    assert event_id in pending_data

    client_two, _embedder_two, dispatcher_two = make_matrix_client(config, vector_store)
    try:
        await client_two.start()

        stored = await wait_until(
            lambda: _event_ids_in_store(vector_store, room_id, {event_id}),
            description="replayed pending event in Qdrant after restart",
        )
        replayed = next(record for record in stored if record.event_id == event_id)
        assert replayed.sender_name == "Alice"

        async def pending_journal_cleared():
            with open(pending_path) as pending_file:
                return json.load(pending_file) == {}

        await wait_until(
            pending_journal_cleared,
            description="pending index journal to clear after replay",
        )
    finally:
        await _stop_client(client_two, dispatcher_two)


async def test_send_message_and_get_message_context_against_real_server(
    matrix_api,
    make_matrix_config,
    make_vector_store,
    make_matrix_client,
):
    alice = await matrix_api.register_user("alice", display_name="Alice")
    bot = await matrix_api.register_user("bot", display_name="Bot")
    room_id = await matrix_api.create_room(alice, name="Context Room")
    await matrix_api.invite_and_join(room_id, alice, bot)

    vector_store = await make_vector_store()
    config = make_matrix_config(bot)
    client, _embedder, dispatcher = make_matrix_client(config, vector_store)

    try:
        await client.start()
        await matrix_api.send_text(room_id, alice, "before one")
        await matrix_api.send_text(room_id, alice, "before two")

        send_result = await client.send_message(room_id, "bot pivot")
        pivot_event_id = send_result["event_id"]
        event = await matrix_api.get_event(room_id, pivot_event_id, alice)
        assert event["content"]["body"] == "bot pivot"

        await matrix_api.send_text(room_id, alice, "after one")
        await matrix_api.send_text(room_id, alice, "after two")

        context = await client.get_message_context(
            room_id=room_id,
            event_id=pivot_event_id,
            before=2,
            after=2,
        )
        assert context["event"]["event_id"] == pivot_event_id
        assert context["event"]["body"] == "bot pivot"
        assert {item["body"] for item in context["before"]} == {"before one", "before two"}
        assert {item["body"] for item in context["after"]} == {"after one", "after two"}
    finally:
        await _stop_client(client, dispatcher)
