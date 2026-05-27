import asyncio
import hashlib
import os
import socket
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Awaitable, Callable
from urllib.parse import quote, urlparse

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from nio_mcp.matrix_client import MatrixMCPClient
from nio_mcp.vector_store import VectorStore
from nio_mcp.webhook import WebhookDispatcher


QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
MATRIX_HOMESERVER_URL = os.environ.get("MATRIX_HOMESERVER_URL", "http://localhost:8008")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as requiring external services"
    )


def _host_and_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme for {url}")
    if parsed.hostname is None:
        raise ValueError(f"Missing hostname in {url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.hostname, port


def _socket_is_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def qdrant_is_reachable() -> bool:
    return _socket_is_reachable(QDRANT_HOST, QDRANT_PORT)


def matrix_is_reachable() -> bool:
    host, port = _host_and_port(MATRIX_HOMESERVER_URL)
    return _socket_is_reachable(host, port)


skip_if_no_qdrant = pytest.mark.skipif(
    not qdrant_is_reachable(),
    reason=f"Qdrant not reachable at {QDRANT_HOST}:{QDRANT_PORT}",
)

skip_if_no_matrix = pytest.mark.skipif(
    not matrix_is_reachable(),
    reason=f"Matrix homeserver not reachable at {MATRIX_HOMESERVER_URL}",
)


@dataclass
class MatrixUser:
    user_id: str
    access_token: str
    device_id: str
    username: str
    password: str


class FakeEmbeddingClient:
    VECTOR_SIZE = 8

    def __init__(self, fail_once_texts: set[str] | None = None) -> None:
        self._fail_once_texts = set(fail_once_texts or set())

    async def embed(self, text: str) -> list[float]:
        return self._embed_text(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        if text in self._fail_once_texts:
            self._fail_once_texts.remove(text)
            raise RuntimeError(f"Intentional embedding failure for {text}")
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [((digest[i] / 255.0) * 2.0) - 1.0 for i in range(self.VECTOR_SIZE)]


class MatrixTestAPI:
    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=10.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def register_user(
        self,
        username_prefix: str,
        *,
        display_name: str | None = None,
    ) -> MatrixUser:
        username = f"{username_prefix}_{uuid.uuid4().hex[:8]}"
        password = f"pw-{uuid.uuid4().hex}"
        device_id = f"DEV{uuid.uuid4().hex[:8].upper()}"
        payload = {
            "username": username,
            "password": password,
            "device_id": device_id,
        }
        response = await self._send("POST", "/_matrix/client/v3/register", json=payload)
        if response.status_code == 401:
            session = response.json()["session"]
            payload["auth"] = {"type": "m.login.dummy", "session": session}
            response = await self._send("POST", "/_matrix/client/v3/register", json=payload)
        response.raise_for_status()
        data = response.json()
        user = MatrixUser(
            user_id=data["user_id"],
            access_token=data["access_token"],
            device_id=data["device_id"],
            username=username,
            password=password,
        )
        if display_name:
            await self.set_display_name(user, display_name)
        return user

    async def set_display_name(self, user: MatrixUser, display_name: str) -> None:
        path = f"/_matrix/client/v3/profile/{quote(user.user_id, safe='')}/displayname"
        await self._request(
            "PUT",
            path,
            access_token=user.access_token,
            json={"displayname": display_name},
        )

    async def create_room(
        self,
        user: MatrixUser,
        *,
        invitees: list[MatrixUser] | None = None,
        name: str | None = None,
    ) -> str:
        payload = {"preset": "private_chat"}
        if invitees:
            payload["invite"] = [invitee.user_id for invitee in invitees]
        if name:
            payload["name"] = name
        data = await self._request(
            "POST",
            "/_matrix/client/v3/createRoom",
            access_token=user.access_token,
            json=payload,
        )
        return data["room_id"]

    async def join_room(self, room_id: str, user: MatrixUser) -> None:
        path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/join"
        await self._request("POST", path, access_token=user.access_token, json={})

    async def invite_user(self, room_id: str, inviter: MatrixUser, invitee: MatrixUser) -> None:
        path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/invite"
        await self._request(
            "POST",
            path,
            access_token=inviter.access_token,
            json={"user_id": invitee.user_id},
        )

    async def invite_and_join(
        self,
        room_id: str,
        inviter: MatrixUser,
        invitee: MatrixUser,
    ) -> None:
        await self.invite_user(room_id, inviter, invitee)
        await self.join_room(room_id, invitee)

    async def send_text(self, room_id: str, user: MatrixUser, body: str) -> str:
        txn_id = uuid.uuid4().hex
        path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/send/m.room.message/{txn_id}"
        data = await self._request(
            "PUT",
            path,
            access_token=user.access_token,
            json={"msgtype": "m.text", "body": body},
        )
        return data["event_id"]

    async def get_event(self, room_id: str, event_id: str, user: MatrixUser) -> dict:
        path = (
            f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/"
            f"{quote(event_id, safe='')}"
        )
        return await self._request("GET", path, access_token=user.access_token)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        access_token: str | None = None,
        **kwargs,
    ) -> dict:
        response = await self._send(
            method,
            path,
            access_token=access_token,
            **kwargs,
        )
        response.raise_for_status()
        if response.content:
            return response.json()
        return {}

    async def _send(
        self,
        method: str,
        path: str,
        *,
        access_token: str | None = None,
        **kwargs,
    ) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        response = None
        for _ in range(20):
            response = await self._client.request(method, path, headers=headers, **kwargs)
            if response.status_code != 429:
                return response
            retry_after_ms = 250
            if response.headers.get("Retry-After"):
                retry_after_ms = int(float(response.headers["Retry-After"]) * 1000)
            else:
                try:
                    retry_after_ms = int(response.json().get("retry_after_ms", retry_after_ms))
                except Exception:
                    pass
            await asyncio.sleep(retry_after_ms / 1000)
        assert response is not None
        return response


async def wait_until(
    predicate: Callable[[], Awaitable[object] | object],
    *,
    timeout: float = 10.0,
    interval: float = 0.1,
    description: str = "condition",
):
    deadline = asyncio.get_running_loop().time() + timeout
    last_error = None
    while True:
        try:
            result = predicate()
            if asyncio.iscoroutine(result):
                result = await result
            if result:
                return result
        except AssertionError as exc:
            last_error = exc
        if asyncio.get_running_loop().time() >= deadline:
            if last_error is not None:
                raise last_error
            raise TimeoutError(f"Timed out waiting for {description}")
        await asyncio.sleep(interval)


@pytest.fixture
async def matrix_api():
    api = MatrixTestAPI(MATRIX_HOMESERVER_URL)
    try:
        yield api
    finally:
        await api.close()


@pytest.fixture
def make_matrix_config(tmp_path):
    def factory(
        user: MatrixUser,
        *,
        store_name: str | None = None,
        backfill_limit: int = 5,
        backfill_pages_max: int = 0,
        message_buffer_size: int = 100,
        matrix_sync_timeout_ms: int = 250,
    ):
        store_dir = tmp_path / (store_name or f"nio_store_{uuid.uuid4().hex[:8]}")
        return SimpleNamespace(
            matrix_homeserver_url=MATRIX_HOMESERVER_URL,
            matrix_access_token=user.access_token,
            matrix_user_id=user.user_id,
            matrix_device_id=user.device_id,
            matrix_store_path=str(store_dir),
            matrix_key_backup_file="",
            matrix_key_backup_passphrase="",
            backfill_limit=backfill_limit,
            backfill_pages_max=backfill_pages_max,
            message_buffer_size=message_buffer_size,
            matrix_sync_timeout_ms=matrix_sync_timeout_ms,
        )

    return factory


@pytest.fixture
async def make_vector_store():
    stores: list[tuple[VectorStore, str]] = []

    async def factory(*, collection: str | None = None) -> VectorStore:
        selected = collection or f"test_{uuid.uuid4().hex[:8]}"
        store = VectorStore(host=QDRANT_HOST, port=QDRANT_PORT, collection=selected)
        await store.init_collection(vector_size=FakeEmbeddingClient.VECTOR_SIZE)
        stores.append((store, selected))
        return store

    try:
        yield factory
    finally:
        cleanup_client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        try:
            for store, collection in stores:
                try:
                    await cleanup_client.delete_collection(collection)
                except Exception:
                    pass
                await store.close()
        finally:
            await cleanup_client.close()


@pytest.fixture
def make_matrix_client():
    def factory(
        config,
        vector_store: VectorStore,
        *,
        embedding_client: FakeEmbeddingClient | None = None,
        webhook_dispatcher: WebhookDispatcher | None = None,
    ) -> tuple[MatrixMCPClient, FakeEmbeddingClient, WebhookDispatcher]:
        embedder = embedding_client or FakeEmbeddingClient()
        dispatcher = webhook_dispatcher or WebhookDispatcher()
        client = MatrixMCPClient(config, vector_store, embedder, dispatcher)
        return client, embedder, dispatcher

    return factory
