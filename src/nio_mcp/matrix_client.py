import asyncio
import logging
import os
from collections import deque
from typing import Optional

from nio import (
    AsyncClient,
    ClientConfig,
    MatrixRoom,
    RoomMessageText,
    SyncResponse,
    RoomMessagesResponse,
    RoomContextResponse,
)
from nio.events.room_events import RoomMessageText as RoomMessageTextEvent
from nio.responses import JoinedRoomsResponse
import nio

from nio_mcp.config import Settings
from nio_mcp.embeddings import EmbeddingClient
from nio_mcp.models import MessageRecord
from nio_mcp.vector_store import VectorStore
from nio_mcp.webhook import WebhookDispatcher

logger = logging.getLogger(__name__)


class MatrixMCPClient:
    def __init__(
        self,
        config: Settings,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        webhook_dispatcher: WebhookDispatcher,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._embedding_client = embedding_client
        self._webhook_dispatcher = webhook_dispatcher
        self._client: Optional[AsyncClient] = None
        self._buffer: deque[MessageRecord] = deque(maxlen=config.message_buffer_size)
        self._seen_event_ids: set[str] = set()
        self._sync_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        os.makedirs(self._config.matrix_store_path, exist_ok=True)

        self._client = AsyncClient(
            homeserver=self._config.matrix_homeserver_url,
            user=self._config.matrix_user_id,
            store_path=self._config.matrix_store_path,
            config=ClientConfig(store_sync_tokens=True),
        )
        self._client.restore_login(
            user_id=self._config.matrix_user_id,
            device_id=self._config.matrix_device_id,
            access_token=self._config.matrix_access_token,
        )

        stored_token = self._client.loaded_sync_token

        if stored_token:
            # Restart: resume from the persisted token; sync_forever delivers all
            # events missed during downtime without a redundant backfill.
            logger.info("Resuming from stored sync token %s", stored_token)
            self._client.add_event_callback(self._on_message, RoomMessageText)
            self._sync_task = asyncio.create_task(
                self._client.sync_forever(
                    since=stored_token,
                    timeout=30000,
                )
            )
        else:
            # Fresh start: anchor position, backfill history, then begin live sync.
            logger.info("Performing initial sync to anchor timeline position")
            initial_sync: SyncResponse = await self._client.sync(full_state=True)

            logger.info("Backfilling room history")
            await self._backfill(initial_sync)

            logger.info("Indexing initial sync timeline events")
            await self._index_initial_sync(initial_sync)

            self._client.add_event_callback(self._on_message, RoomMessageText)

            logger.info("Starting live sync from token %s", initial_sync.next_batch)
            self._sync_task = asyncio.create_task(
                self._client.sync_forever(
                    since=initial_sync.next_batch,
                    timeout=30000,
                )
            )

    async def stop(self) -> None:
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.close()

    async def get_recent_messages(
        self,
        k: int = 20,
        sender: Optional[str] = None,
        room_id: Optional[str] = None,
    ) -> list[MessageRecord]:
        results = list(self._buffer)
        if sender:
            results = [m for m in results if m.sender == sender]
        if room_id:
            results = [m for m in results if m.room_id == room_id]
        results.sort(key=lambda m: m.timestamp)
        return results[-k:]

    async def send_message(
        self,
        room_id: str,
        body: str,
        msgtype: str = "m.text",
    ) -> dict:
        if self._client is None:
            raise RuntimeError("Client not started")
        response = await self._client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": msgtype, "body": body},
        )
        if isinstance(response, nio.RoomSendResponse):
            return {"event_id": response.event_id}
        return {"error": str(response)}

    async def get_message_context(
        self,
        room_id: str,
        event_id: str,
        before: int = 5,
        after: int = 5,
    ) -> dict:
        if self._client is None:
            raise RuntimeError("Client not started")
        response: RoomContextResponse = await self._client.room_context(
            room_id=room_id,
            event_id=event_id,
            limit=before + after,
        )
        if isinstance(response, nio.ErrorResponse):
            return {"error": str(response)}
        return {
            "event": self._event_to_dict(room_id, response.event) if response.event else None,
            "before": [self._event_to_dict(room_id, e) for e in (response.events_before or [])],
            "after": [self._event_to_dict(room_id, e) for e in (response.events_after or [])],
        }

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    async def _index_initial_sync(self, initial_sync: SyncResponse) -> None:
        if not hasattr(initial_sync, "rooms") or not initial_sync.rooms:
            return
        joined = getattr(initial_sync.rooms, "join", {})
        records: list[MessageRecord] = []
        for room_id, room_info in joined.items():
            timeline = getattr(room_info, "timeline", None)
            if timeline is None:
                continue
            for event in getattr(timeline, "events", []) or []:
                if isinstance(event, RoomMessageText):
                    record = self._parse_event(room_id, event)
                    if record and record.event_id not in self._seen_event_ids:
                        self._seen_event_ids.add(record.event_id)
                        self._buffer.append(record)
                        records.append(record)
        if records:
            await self._batch_index(records)
        logger.info("Indexed %d messages from initial sync timeline", len(records))

    async def _backfill(self, initial_sync: SyncResponse) -> None:
        rooms_response: JoinedRoomsResponse = await self._client.joined_rooms()
        room_ids = rooms_response.rooms if hasattr(rooms_response, "rooms") else []

        all_records: list[MessageRecord] = []
        for room_id in room_ids:
            # Get the prev_batch token for this room from the initial sync
            prev_batch: Optional[str] = None
            if hasattr(initial_sync, "rooms") and initial_sync.rooms:
                joined = getattr(initial_sync.rooms, "join", {})
                if room_id in joined:
                    timeline = getattr(joined[room_id], "timeline", None)
                    if timeline:
                        prev_batch = getattr(timeline, "prev_batch", None)

            if prev_batch is None:
                logger.debug("No prev_batch for room %s, skipping backfill", room_id)
                continue

            all_records.extend(await self._backfill_room(room_id, prev_batch))

        # Insert oldest-first so that when the deque is full, oldest records are
        # evicted naturally as newer messages arrive — never the newest ones.
        all_records.sort(key=lambda m: m.timestamp)
        for record in all_records:
            self._buffer.append(record)

    async def _backfill_room(self, room_id: str, prev_batch: str) -> list[MessageRecord]:
        pages_max = self._config.backfill_pages_max  # 0 = unlimited
        page = 0
        fetched = 0
        all_records: list[MessageRecord] = []

        while pages_max == 0 or page < pages_max:
            response: RoomMessagesResponse = await self._client.room_messages(
                room_id=room_id,
                start=prev_batch,
                direction=nio.MessageDirection.back,
                limit=self._config.backfill_limit,
            )
            if isinstance(response, nio.ErrorResponse):
                logger.warning("room_messages error for %s: %s", room_id, response)
                break

            page_records: list[MessageRecord] = []
            for event in response.chunk:
                if isinstance(event, RoomMessageText):
                    record = self._parse_event(room_id, event)
                    if record and record.event_id not in self._seen_event_ids:
                        page_records.append(record)
                        self._seen_event_ids.add(record.event_id)

            if page_records:
                await self._batch_index(page_records)
                all_records.extend(page_records)

            fetched += len(response.chunk)
            page += 1

            # Spec-correct termination: end absent means no more pages
            if response.end is None:
                break
            prev_batch = response.end

        logger.info("Backfilled %d messages from room %s (%d pages)", fetched, room_id, page)
        return all_records

    async def _on_message(self, room: MatrixRoom, event: RoomMessageTextEvent) -> None:
        if event.event_id in self._seen_event_ids:
            return
        self._seen_event_ids.add(event.event_id)
        member = room.users.get(event.sender)
        sender_name = (member.display_name if member and member.display_name else None) or self._sender_display_name(event.sender)
        record = MessageRecord(
            event_id=event.event_id,
            room_id=room.room_id,
            sender=event.sender,
            sender_name=sender_name,
            body=event.body,
            timestamp=event.server_timestamp,
        )
        self._buffer.append(record)
        await self._index_message(record)

    async def _index_message(self, record: MessageRecord) -> None:
        try:
            vector = await self._embedding_client.embed(f"{record.sender_name}: {record.body}")
            await self._vector_store.upsert(record, vector)
            await self._webhook_dispatcher.dispatch(record)
        except Exception:
            logger.exception("Failed to index message %s", record.event_id)

    async def _batch_index(self, records: list[MessageRecord]) -> None:
        if not records:
            return
        try:
            texts = [f"{r.sender_name}: {r.body}" for r in records]
            vectors = await self._embedding_client.embed_batch(texts)
            for record, vector in zip(records, vectors):
                await self._vector_store.upsert(record, vector)
        except Exception:
            logger.exception("Batch indexing failed for %d records", len(records))

    def _sender_display_name(self, sender: str) -> str:
        if sender.startswith("@") and ":" in sender:
            return sender[1:sender.index(":")]
        return sender

    def _resolve_display_name(self, room_id: str, sender: str) -> str:
        if self._client:
            room = self._client.rooms.get(room_id)
            if room:
                member = room.users.get(sender)
                if member and member.display_name:
                    return member.display_name
        return self._sender_display_name(sender)

    def _parse_event(self, room_id: str, event: RoomMessageTextEvent) -> Optional[MessageRecord]:
        body = getattr(event, "body", None)
        if not body:
            return None
        return MessageRecord(
            event_id=event.event_id,
            room_id=room_id,
            sender=event.sender,
            sender_name=self._resolve_display_name(room_id, event.sender),
            body=body,
            timestamp=event.server_timestamp,
        )

    @staticmethod
    def _event_to_dict(room_id: str, event) -> dict:
        return {
            "event_id": getattr(event, "event_id", None),
            "room_id": room_id,
            "sender": getattr(event, "sender", None),
            "body": getattr(event, "body", None),
            "timestamp": getattr(event, "server_timestamp", None),
        }
