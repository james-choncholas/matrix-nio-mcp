import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Optional

import httpx

from nio_mcp.models import MessageRecord

logger = logging.getLogger(__name__)


def _record_to_json(record: MessageRecord) -> str:
    return json.dumps(record.to_dict())


class WebhookDispatcher:
    def __init__(
        self,
        webhook_url: str = "",
        webhook_secret: str = "",
        queue_maxsize: int = 100,
    ) -> None:
        self._webhook_url = webhook_url
        self._webhook_secret = webhook_secret
        self._queue_maxsize = queue_maxsize
        self._subscribers: set[asyncio.Queue] = set()
        self._http: Optional[httpx.AsyncClient] = None

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=10.0)
        return self._http

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
            await self._http_post(record, payload)

    async def _http_post(self, record: MessageRecord, payload: str) -> None:
        headers = {"Content-Type": "application/json"}
        if self._webhook_secret:
            sig = hmac.new(
                self._webhook_secret.encode(),
                payload.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Nio-MCP-Signature"] = f"sha256={sig}"
        try:
            async with self._get_http() as client:
                resp = await client.post(self._webhook_url, content=payload, headers=headers)
                resp.raise_for_status()
        except Exception as exc:
            logger.error("Webhook POST to %s failed: %s", self._webhook_url, exc)

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()
