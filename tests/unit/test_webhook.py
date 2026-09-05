import asyncio
import json
from typing import Any, Mapping

import pytest

from nio_mcp.models import MessageRecord
from nio_mcp.webhook import WebhookDispatcher, _render_per_msg, _render_prompt


RECORD = MessageRecord(
    event_id="$abc:example.org",
    room_id="!room:example.org",
    room_name="Test Room",
    sender="@alice:example.org",
    sender_name="Alice",
    body="Hello",
    timestamp=1700000000000,
)

RECORD2 = MessageRecord(
    event_id="$def:example.org",
    room_id="!room:example.org",
    room_name="Test Room",
    sender="@bob:example.org",
    sender_name="Bob",
    body="World",
    timestamp=1700000001000,
)


class FakeLLMClient:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.started = False
        self.closed = False
        self.calls: list[dict[str, Any]] = []

    async def start(self) -> None:
        self.started = True

    async def run(
        self,
        *,
        model: str,
        prompt: str,
        request_options: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "request_options": dict(request_options),
            }
        )
        if self.error is not None:
            raise self.error
        return {
            "content": "no relevant messages received, passing for now",
            "output": [{"type": "tool_call", "name": "example"}],
        }

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def dispatcher():
    return WebhookDispatcher(queue_maxsize=3)


# -- SSE subscriber mechanics -------------------------------------------------

def test_subscribe_returns_bounded_queue(dispatcher):
    queue = dispatcher.subscribe()
    assert queue.maxsize == 3
    assert queue in dispatcher._subscribers


def test_unsubscribe_removes_queue(dispatcher):
    queue = dispatcher.subscribe()
    dispatcher.unsubscribe(queue)
    assert queue not in dispatcher._subscribers


def test_unsubscribe_unknown_queue_is_safe(dispatcher):
    dispatcher.unsubscribe(asyncio.Queue())


async def test_dispatch_delivers_to_all_subscribers(dispatcher):
    queue_1 = dispatcher.subscribe()
    queue_2 = dispatcher.subscribe()
    await dispatcher.dispatch(RECORD)
    assert json.loads(queue_1.get_nowait())["event_id"] == RECORD.event_id
    assert json.loads(queue_2.get_nowait())["event_id"] == RECORD.event_id


async def test_dispatch_does_not_deliver_to_unsubscribed(dispatcher):
    queue = dispatcher.subscribe()
    dispatcher.unsubscribe(queue)
    await dispatcher.dispatch(RECORD)
    assert queue.empty()


async def test_dispatch_full_queue_drops_oldest_not_newest(dispatcher):
    queue = dispatcher.subscribe()
    for index in range(3):
        queue.put_nowait(json.dumps({"body": f"old-{index}"}))
    await dispatcher.dispatch(RECORD)
    items = []
    while not queue.empty():
        items.append(json.loads(queue.get_nowait()))
    assert len(items) == 3
    assert items[-1]["event_id"] == RECORD.event_id


async def test_dispatch_does_not_schedule_callback_without_client(dispatcher):
    queue = dispatcher.subscribe()
    await dispatcher.dispatch(RECORD)
    assert not queue.empty()
    assert dispatcher._cooldown_task is None


# -- Prompt rendering ---------------------------------------------------------

def test_render_per_msg_all_placeholders():
    result = _render_per_msg(
        "{sender_name} ({sender}) in {room_name} ({room}): {message}", RECORD
    )
    assert result == "Alice (@alice:example.org) in Test Room (!room:example.org): Hello"


def test_render_per_msg_subset_of_placeholders():
    assert _render_per_msg("{sender_name} said {message}", RECORD) == "Alice said Hello"


def test_render_per_msg_body_with_braces_not_reinterpreted():
    record = MessageRecord(
        event_id="$x",
        room_id="!r",
        room_name="R",
        sender="@a",
        sender_name="A",
        body="use {sender} carefully",
        timestamp=0,
    )
    assert _render_per_msg("msg: {message}", record) == "msg: use {sender} carefully"


def test_render_prompt_header_prepended_once():
    assert _render_prompt("Header:", "{message}", [RECORD, RECORD2]).splitlines() == [
        "Header:",
        "Hello",
        "World",
    ]


def test_render_prompt_no_header():
    assert _render_prompt("", "{message}", [RECORD, RECORD2]) == "Hello\nWorld"


def test_render_prompt_single_message():
    assert _render_prompt("Hdr:", "{sender_name}: {message}", [RECORD]) == (
        "Hdr:\nAlice: Hello"
    )


def test_render_prompt_per_msg_applied_to_each_record():
    assert _render_prompt("", "{sender_name}", [RECORD, RECORD2]) == "Alice\nBob"


# -- LLM callback dispatch ----------------------------------------------------

async def test_llm_client_receives_rendered_prompt_and_model():
    client = FakeLLMClient()
    dispatcher = WebhookDispatcher(
        llm_client=client,
        prompt_header="Messages:",
        prompt_per_msg="{message}",
        model="steven",
        cooldown_seconds=0.01,
    )
    await dispatcher.start()
    await dispatcher.dispatch(RECORD)
    await asyncio.sleep(0.05)

    assert client.started
    assert client.calls == [
        {
            "model": "steven",
            "prompt": "Messages:\nHello",
            "request_options": {},
        }
    ]


async def test_start_and_close_are_delegated_to_llm_client():
    client = FakeLLMClient()
    dispatcher = WebhookDispatcher(llm_client=client)
    await dispatcher.start()
    await dispatcher.close()
    assert client.started
    assert client.closed


async def test_llm_batches_multiple_messages_in_one_call():
    client = FakeLLMClient()
    dispatcher = WebhookDispatcher(
        llm_client=client,
        prompt_header="",
        prompt_per_msg="{message}",
        cooldown_seconds=0.01,
    )
    await dispatcher.start()
    await dispatcher.dispatch(RECORD)
    await dispatcher.dispatch(RECORD2)
    await asyncio.sleep(0.05)

    assert len(client.calls) == 1
    assert client.calls[0]["prompt"] == "Hello\nWorld"


async def test_cooldown_resets_on_new_message():
    client = FakeLLMClient()
    dispatcher = WebhookDispatcher(llm_client=client, cooldown_seconds=0.05)
    await dispatcher.start()
    await dispatcher.dispatch(RECORD)
    first_task = dispatcher._cooldown_task
    await asyncio.sleep(0.02)
    assert not client.calls
    await dispatcher.dispatch(RECORD2)
    assert first_task is not dispatcher._cooldown_task
    await asyncio.sleep(0.02)
    assert not client.calls
    await asyncio.sleep(0.06)
    assert client.calls


async def test_max_queue_seconds_bounds_wait_under_continuous_traffic():
    # Regression: a pure debounce never fires when messages keep arriving within
    # the cooldown window. The max-queue deadline must force a flush regardless.
    client = FakeLLMClient()
    dispatcher = WebhookDispatcher(
        llm_client=client,
        prompt_header="",
        prompt_per_msg="{message}",
        cooldown_seconds=0.1,   # each message alone would push the timer out 0.1s
        max_queue_seconds=0.2,  # ...but nothing may wait longer than 0.2s
    )
    await dispatcher.start()
    # Seven messages spaced 0.04s apart (< cooldown) => ~0.28s of steady traffic.
    for _ in range(7):
        await dispatcher.dispatch(RECORD)
        await asyncio.sleep(0.04)
    assert client.calls, "batch must flush at the max-queue deadline despite resets"
    await dispatcher.close()


async def test_max_queue_seconds_does_not_flush_before_deadline():
    # Within a single cooldown window and below the max-queue deadline, the batch
    # should still be buffered (normal debounce behaviour is preserved).
    client = FakeLLMClient()
    dispatcher = WebhookDispatcher(
        llm_client=client,
        cooldown_seconds=0.05,
        max_queue_seconds=10.0,
    )
    await dispatcher.start()
    await dispatcher.dispatch(RECORD)
    await asyncio.sleep(0.02)
    assert not client.calls
    await asyncio.sleep(0.05)
    assert client.calls
    await dispatcher.close()


def test_warns_when_max_queue_seconds_below_cooldown(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="nio_mcp.webhook"):
        WebhookDispatcher(cooldown_seconds=300.0, max_queue_seconds=60.0)
    assert any(
        "max_queue_seconds" in record.getMessage()
        and "less than" in record.getMessage().lower()
        for record in caplog.records
    )


def test_no_warning_when_max_queue_seconds_at_least_cooldown(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="nio_mcp.webhook"):
        WebhookDispatcher(cooldown_seconds=60.0, max_queue_seconds=60.0)
        WebhookDispatcher(cooldown_seconds=60.0, max_queue_seconds=120.0)
    assert not any(
        "max_queue_seconds" in record.getMessage() for record in caplog.records
    )


async def test_batch_cap_fires_at_50_without_waiting_for_cooldown():
    client = FakeLLMClient()
    dispatcher = WebhookDispatcher(
        llm_client=client,
        prompt_header="",
        prompt_per_msg="{message}",
        cooldown_seconds=60,
    )
    await dispatcher.start()
    for _ in range(50):
        await dispatcher.dispatch(RECORD)
    await asyncio.sleep(0.05)

    assert len(client.calls) == 1
    assert len(client.calls[0]["prompt"].splitlines()) == 50
    assert dispatcher._pending_records == []
    await dispatcher.close()


async def test_llm_failure_does_not_escape_background_delivery():
    client = FakeLLMClient(error=ConnectionError("refused"))
    dispatcher = WebhookDispatcher(llm_client=client, cooldown_seconds=0.01)
    await dispatcher.start()
    await dispatcher.dispatch(RECORD)
    await asyncio.sleep(0.05)
    assert len(client.calls) == 1


async def test_llm_failure_still_delivers_to_sse_subscribers():
    client = FakeLLMClient(error=ConnectionError("refused"))
    dispatcher = WebhookDispatcher(llm_client=client, cooldown_seconds=0.01)
    await dispatcher.start()
    queue = dispatcher.subscribe()
    await dispatcher.dispatch(RECORD)
    assert json.loads(queue.get_nowait())["event_id"] == RECORD.event_id
    await asyncio.sleep(0.05)


async def test_llm_success_logs_final_message_and_tool_output(caplog):
    import logging

    client = FakeLLMClient()
    dispatcher = WebhookDispatcher(llm_client=client, cooldown_seconds=0.01)
    await dispatcher.start()
    with caplog.at_level(logging.INFO, logger="nio_mcp.webhook"):
        await dispatcher.dispatch(RECORD)
        await asyncio.sleep(0.05)

    compact_messages = [record.message.replace(" ", "") for record in caplog.records]
    assert any(
        '"content":"norelevantmessagesreceived,passingfornow"' in message
        and '"type":"tool_call"' in message
        for message in compact_messages
    )


async def test_llm_failure_logs_warning(caplog):
    import logging

    client = FakeLLMClient(error=ConnectionError("refused"))
    dispatcher = WebhookDispatcher(llm_client=client, cooldown_seconds=0.01)
    await dispatcher.start()
    with caplog.at_level(logging.WARNING, logger="nio_mcp.webhook"):
        await dispatcher.dispatch(RECORD)
        await asyncio.sleep(0.05)
    assert any("llm webhook call failed" in record.message.lower() for record in caplog.records)


async def test_llm_passes_configured_request_options():
    client = FakeLLMClient()
    dispatcher = WebhookDispatcher(
        llm_client=client,
        cooldown_seconds=0.01,
        tools='{"tool_ids": ["server:mcp:matrix"]}',
    )
    await dispatcher.start()
    await dispatcher.dispatch(RECORD)
    await asyncio.sleep(0.05)
    assert client.calls[0]["request_options"] == {
        "tool_ids": ["server:mcp:matrix"]
    }


def test_webhook_tools_must_be_a_json_object():
    with pytest.raises(ValueError, match="JSON object"):
        WebhookDispatcher(tools='["not", "an", "object"]')
