import asyncio
import json
import logging
import re
from typing import Any, Optional

from nio_mcp.llm_callback import LLMCallbackClient
from nio_mcp.models import MessageRecord

_PER_MSG_RE = re.compile(r"\{(message|sender_name|sender|room_name|room)\}")

logger = logging.getLogger(__name__)

_MAX_BATCH_SIZE = 50


def _record_to_json(record: MessageRecord) -> str:
    return json.dumps(record.to_dict())


def _render_per_msg(template: str, record: MessageRecord) -> str:
    """Expand per-message placeholders for a single record.

    Placeholders: {message}, {sender_name}, {sender}, {room_name}, {room}
    Single-pass regex sub so braces inside message bodies are never re-interpreted.
    """
    replacements = {
        "message": record.body,
        "sender_name": record.sender_name,
        "sender": record.sender,
        "room_name": record.room_name,
        "room": record.room_id,
    }
    return _PER_MSG_RE.sub(lambda m: replacements[m.group(1)], template)


def _render_prompt(header: str, per_msg_template: str, records: list[MessageRecord]) -> str:
    """Build the full LLM user message: header (once) then per-message lines."""
    lines = [_render_per_msg(per_msg_template, r) for r in records]
    if header:
        return header + "\n" + "\n".join(lines)
    return "\n".join(lines)


class WebhookDispatcher:
    def __init__(
        self,
        llm_client: LLMCallbackClient | None = None,
        prompt_header: str = "New Matrix messages:",
        prompt_per_msg: str = "{sender_name} ({sender}) in {room_name} ({room}): {message}",
        model: str = "gpt-4o-mini",
        cooldown_seconds: float = 300.0,
        max_queue_seconds: float = 900.0,
        queue_maxsize: int = 100,
        tools: str = "",
    ) -> None:
        self._llm_client = llm_client
        self._prompt_header = prompt_header
        self._prompt_per_msg = prompt_per_msg
        self._model = model
        self._cooldown_seconds = cooldown_seconds
        self._max_queue_seconds = max_queue_seconds
        if max_queue_seconds < cooldown_seconds:
            logger.warning(
                "webhook max_queue_seconds (%.1f) is less than cooldown_seconds "
                "(%.1f); the maximum queue age will cap the cooldown, so batches "
                "fire after %.1fs of buffering instead of after a quiet period",
                max_queue_seconds,
                cooldown_seconds,
                max_queue_seconds,
            )
        self._queue_maxsize = queue_maxsize
        self._request_options: dict[str, Any] = json.loads(tools) if tools else {}
        if not isinstance(self._request_options, dict):
            raise ValueError("WEBHOOK_TOOLS must contain a JSON object")
        self._subscribers: set[asyncio.Queue] = set()
        self._pending_records: list[MessageRecord] = []
        self._cooldown_task: Optional[asyncio.Task] = None
        self._delivery_tasks: set[asyncio.Task] = set()
        # Monotonic timestamp of the oldest un-delivered record in the current
        # batch. Used to enforce an upper bound on how long any message waits.
        self._batch_started_at: Optional[float] = None

    async def start(self) -> None:
        if self._llm_client is not None:
            await self._llm_client.start()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def dispatch(self, record: MessageRecord) -> None:
        payload = _record_to_json(record)
        for q in list(self._subscribers):  # snapshot avoids mutation during iteration
            if q.full():
                try:
                    q.get_nowait()
                    logger.warning(
                        "SSE subscriber queue full; dropping oldest event for room %s",
                        record.room_id,
                    )
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(payload)

        if self._llm_client is not None:
            self._pending_records.append(record)
            now = asyncio.get_running_loop().time()
            if self._batch_started_at is None:
                self._batch_started_at = now
            if self._cooldown_task is not None and not self._cooldown_task.done():
                self._cooldown_task.cancel()
            if len(self._pending_records) >= _MAX_BATCH_SIZE:
                records = self._pending_records[:_MAX_BATCH_SIZE]
                del self._pending_records[:_MAX_BATCH_SIZE]
                self._cooldown_task = None
                self._batch_started_at = now if self._pending_records else None
                self._schedule_delivery(records)
            else:
                # Debounce for the cooldown, but never let the oldest buffered
                # message wait longer than max_queue_seconds. Capping the sleep
                # at the remaining time until that deadline guarantees the batch
                # flushes even if messages keep arriving within the cooldown.
                deadline = self._batch_started_at + self._max_queue_seconds
                wait = min(self._cooldown_seconds, deadline - now)
                self._cooldown_task = asyncio.create_task(self._cooldown_fire(wait))

    async def _cooldown_fire(self, wait: float) -> None:
        try:
            if wait > 0:
                await asyncio.sleep(wait)
        except asyncio.CancelledError:
            return
        records, self._pending_records = self._pending_records, []
        self._cooldown_task = None
        self._batch_started_at = None
        if not records:
            return
        await self._deliver(records)

    def _schedule_delivery(self, records: list[MessageRecord]) -> None:
        task = asyncio.create_task(self._deliver(records))
        self._delivery_tasks.add(task)
        task.add_done_callback(self._delivery_tasks.discard)

    async def _deliver(self, records: list[MessageRecord]) -> None:
        try:
            await self._call_llm(records)
        except Exception:
            logger.warning(
                "LLM webhook call failed for %d buffered message(s); continuing",
                len(records),
                exc_info=True,
            )

    async def _call_llm(self, records: list[MessageRecord]) -> None:
        if self._llm_client is None:
            return
        content = _render_prompt(self._prompt_header, self._prompt_per_msg, records)
        result = await self._llm_client.run(
            model=self._model,
            prompt=content,
            request_options=self._request_options,
        )
        logger.info(
            "LLM webhook response: messages=%d model=%s body=%s",
            len(records),
            self._model,
            json.dumps(result, ensure_ascii=False),
        )
        logger.debug(
            "LLM webhook: completed call with %d message(s); model=%s",
            len(records),
            self._model,
        )

    async def close(self) -> None:
        if self._cooldown_task is not None and not self._cooldown_task.done():
            self._cooldown_task.cancel()
        for task in self._delivery_tasks:
            task.cancel()
        if self._llm_client is not None:
            await self._llm_client.close()
