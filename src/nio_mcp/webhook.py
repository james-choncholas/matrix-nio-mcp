import asyncio
import json
import logging
import re
from typing import Optional

import httpx

from nio_mcp.models import MessageRecord

_PER_MSG_RE = re.compile(r"\{(message|sender_name|sender|room_name|room)\}")

logger = logging.getLogger(__name__)


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
        webhook_url: str = "",
        bearer_token: str = "",
        prompt_header: str = "New Matrix messages:",
        prompt_per_msg: str = "{sender_name} ({sender}) in {room_name} ({room}): {message}",
        model: str = "gpt-4o-mini",
        cooldown_seconds: float = 300.0,
        queue_maxsize: int = 100,
    ) -> None:
        self._webhook_url = webhook_url.rstrip("/")
        self._bearer_token = bearer_token
        self._prompt_header = prompt_header
        self._prompt_per_msg = prompt_per_msg
        self._model = model
        self._cooldown_seconds = cooldown_seconds
        self._queue_maxsize = queue_maxsize
        self._subscribers: set[asyncio.Queue] = set()
        self._http: Optional[httpx.AsyncClient] = None
        self._pending_records: list[MessageRecord] = []
        self._cooldown_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._http = httpx.AsyncClient(timeout=30.0)

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

        if self._webhook_url:
            self._pending_records.append(record)
            if self._cooldown_task is not None and not self._cooldown_task.done():
                self._cooldown_task.cancel()
            self._cooldown_task = asyncio.create_task(self._cooldown_fire())

    async def _cooldown_fire(self) -> None:
        try:
            await asyncio.sleep(self._cooldown_seconds)
        except asyncio.CancelledError:
            return
        records, self._pending_records = self._pending_records, []
        self._cooldown_task = None
        if not records:
            return
        try:
            await self._call_llm(records)
        except Exception:
            logger.warning(
                "LLM webhook call failed for %d buffered message(s); continuing",
                len(records),
                exc_info=True,
            )

    async def _call_llm(self, records: list[MessageRecord]) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30.0)
        content = _render_prompt(self._prompt_header, self._prompt_per_msg, records)
        url = f"{self._webhook_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
        }
        resp = await self._http.post(url, json=body, headers=headers)
        resp.raise_for_status()
        logger.debug(
            "LLM webhook: called with %d message(s); model=%s status=%d",
            len(records),
            self._model,
            resp.status_code,
        )

    async def close(self) -> None:
        if self._cooldown_task is not None and not self._cooldown_task.done():
            self._cooldown_task.cancel()
        if self._http and not self._http.is_closed:
            await self._http.aclose()
