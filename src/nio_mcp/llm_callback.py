"""Pluggable LLM callback clients.

The webhook dispatcher deliberately depends only on :class:`LLMCallbackClient`.
OpenWebUI's current chat-based agent loop is implemented here so a future
Responses API client can replace it without changing batching or Matrix code.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Mapping, Protocol

import httpx

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 0.5
_PROTECTED_REQUEST_FIELDS = {
    "background_tasks",
    "chat_id",
    "id",
    "messages",
    "model",
    "stream",
}


class LLMCallbackClient(Protocol):
    """Backend-independent interface used by ``WebhookDispatcher``."""

    async def start(self) -> None: ...

    async def run(
        self,
        *,
        model: str,
        prompt: str,
        request_options: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class OpenWebUIChatClient:
    """Run a prompt through OpenWebUI's native server-side tool loop.

    OpenWebUI currently executes its multi-round native tool loop only for a
    streamed completion attached to a persisted chat message. This client owns
    that protocol: it creates a chat, starts the completion, and reads the
    final assistant message. The chat is left in OpenWebUI's UI afterward.

    ``base_url`` accepts either the OpenWebUI origin (``https://host``) or the
    existing OpenAI-compatible setting (``https://host/api/v1``).
    """

    def __init__(
        self,
        *,
        base_url: str,
        bearer_token: str = "",
        timeout_seconds: float = 300.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_root = _openwebui_api_root(base_url)
        self._headers = {"Content-Type": "application/json"}
        if bearer_token:
            self._headers["Authorization"] = f"Bearer {bearer_token}"
        self._timeout = httpx.Timeout(
            connect=10.0, read=timeout_seconds, write=30.0, pool=10.0
        )
        self._run_timeout_seconds = timeout_seconds
        self._http = http_client
        self._owns_http = http_client is None

    async def start(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)

    async def run(
        self,
        *,
        model: str,
        prompt: str,
        request_options: Mapping[str, Any],
    ) -> dict[str, Any]:
        conflicts = _PROTECTED_REQUEST_FIELDS.intersection(request_options)
        if conflicts:
            fields = ", ".join(sorted(conflicts))
            raise ValueError(f"LLM request options cannot override: {fields}")

        await self.start()
        assert self._http is not None

        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        timestamp = int(time.time())

        create_response = await self._request(
            "POST",
            "/v1/chats/new",
            json={
                "chat": {
                    "title": "nio-mcp callback",
                    "models": [model],
                    "history": {
                        "currentId": assistant_message_id,
                        "messages": {
                            user_message_id: {
                                "id": user_message_id,
                                "role": "user",
                                "content": prompt,
                                "timestamp": timestamp,
                                "models": [model],
                                "childrenIds": [assistant_message_id],
                            },
                            assistant_message_id: {
                                "id": assistant_message_id,
                                "role": "assistant",
                                "content": "",
                                "parentId": user_message_id,
                                "childrenIds": [],
                                "model": model,
                                "modelName": model,
                                "modelIdx": 0,
                                "done": False,
                                "timestamp": timestamp + 1,
                            },
                        },
                    },
                }
            },
        )
        create_payload = create_response.json()
        if not isinstance(create_payload, dict) or not isinstance(
            create_payload.get("id"), str
        ):
            raise ValueError("OpenWebUI create-chat response did not contain an id")
        chat_id = create_payload["id"]

        completion_body = {
            **request_options,
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "chat_id": chat_id,
            "id": assistant_message_id,
            "background_tasks": {
                "title_generation": False,
                "tags_generation": False,
                "follow_up_generation": False,
            },
        }
        await self._request("POST", "/chat/completions", json=completion_body)

        # A session_id makes OpenWebUI run the request asynchronously. The
        # normal MCP-only callback omits it and blocks in the POST above.
        if request_options.get("session_id"):
            await self._wait_for_tasks(chat_id)

        final_response = await self._request("GET", f"/v1/chats/{chat_id}")
        return _assistant_message(final_response.json(), assistant_message_id)

    async def _wait_for_tasks(self, chat_id: str) -> None:
        deadline = asyncio.get_running_loop().time() + self._run_timeout_seconds
        while True:
            response = await self._request("GET", f"/tasks/chat/{chat_id}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("OpenWebUI task response was not an object")
            task_ids = payload.get("task_ids", [])
            if not task_ids:
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"OpenWebUI chat {chat_id} did not finish within "
                    f"{self._run_timeout_seconds:g}s"
                )
            await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, remaining))

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        assert self._http is not None
        response = await self._http.request(
            method,
            f"{self._api_root}{path}",
            headers=self._headers,
            json=json,
        )
        if not response.is_success:
            logger.warning(
                "OpenWebUI callback request failed: method=%s path=%s status=%d body=%s",
                method,
                path,
                response.status_code,
                response.text,
            )
        response.raise_for_status()
        return response

    async def close(self) -> None:
        if self._owns_http and self._http is not None:
            if not self._http.is_closed:
                await self._http.aclose()
            self._http = None


def create_llm_callback_client(
    *,
    backend: str,
    base_url: str,
    bearer_token: str,
    timeout_seconds: float,
) -> LLMCallbackClient | None:
    """Construct the configured backend at the application composition root."""
    if not base_url:
        return None
    if backend == "openwebui_chat":
        return OpenWebUIChatClient(
            base_url=base_url,
            bearer_token=bearer_token,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported LLM callback backend: {backend}")


def _openwebui_api_root(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/api/v1"):
        return base_url.removesuffix("/v1")
    if base_url.endswith("/api"):
        return base_url
    return f"{base_url}/api"


def _assistant_message(payload: Any, message_id: str) -> dict[str, Any]:
    try:
        message = payload["chat"]["history"]["messages"][message_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"OpenWebUI chat response did not contain assistant message {message_id}"
        ) from exc
    if not isinstance(message, dict):
        raise ValueError("OpenWebUI assistant message was not an object")
    error = message.get("error")
    if error:
        if isinstance(error, dict):
            detail = error.get("content") or error.get("message") or str(error)
        else:
            detail = str(error)
        raise RuntimeError(f"OpenWebUI assistant message failed: {detail}")
    return message
