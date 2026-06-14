import asyncio
import hashlib
import logging
import os
from typing import Optional

from nio import (
    AsyncClient,
    AsyncClientConfig,
    MatrixRoom,
    RoomMessageText,
    RoomMemberEvent,
    RoomNameEvent,
    RoomAliasEvent,
    SyncResponse,
    RoomMessagesResponse,
    RoomContextResponse,
)
from nio.responses import JoinedRoomsResponse
import nio

from nio_mcp.config import Settings
from nio_mcp.embeddings import EmbeddingClient
from nio_mcp.models import MessageRecord
from nio_mcp.store import MessageStore
from nio_mcp.vector_store import VectorStore
from nio_mcp.webhook import WebhookDispatcher

logger = logging.getLogger(__name__)

_META_BACKFILL_COMPLETE = "backfill_complete"
_META_KEY_BACKUP = "key_backup_imported"


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
        self._sync_task: Optional[asyncio.Task] = None
        self._store = MessageStore(
            os.path.join(config.matrix_store_path, "nio_mcp.db")
        )

    async def start(self) -> None:
        os.makedirs(self._config.matrix_store_path, exist_ok=True)
        self._store.open()

        self._client = AsyncClient(
            homeserver=self._config.matrix_homeserver_url,
            user=self._config.matrix_user_id,
            store_path=self._config.matrix_store_path,
            config=AsyncClientConfig(store_sync_tokens=True),
        )
        self._client.restore_login(
            user_id=self._config.matrix_user_id,
            device_id=self._config.matrix_device_id,
            access_token=self._config.matrix_access_token,
        )

        key_just_imported = await self._import_key_backup()

        stored_token = self._client.loaded_sync_token

        if key_just_imported and self._is_backfill_complete():
            # A key backup was imported for the first time on this run.  During the
            # previous backfill (which ran without the key) encrypted messages came
            # back from nio as MegolmEvent, not RoomMessageText, so they were silently
            # skipped.  Clearing the sentinel forces a full re-backfill so those
            # messages are decrypted and indexed with the newly available session keys.
            logger.info(
                "E2EE key backup imported for the first time; clearing backfill sentinel "
                "so historical encrypted messages are re-indexed with the new session keys"
            )
            self._store.delete_meta(_META_BACKFILL_COMPLETE)

        if stored_token and self._is_backfill_complete():
            # Restart: rooms are in the DB so send_message works immediately; resume
            # live sync from the stored token without re-running backfill.
            self._restore_rooms_to_client()
            await self._retry_pending_index()
            logger.info("Resuming from stored sync token %s", stored_token)
            self._client.add_event_callback(self._on_message, RoomMessageText)
            self._client.add_event_callback(self._on_room_name, RoomNameEvent)
            self._client.add_event_callback(self._on_room_name, RoomAliasEvent)
            self._client.add_event_callback(self._on_room_member, RoomMemberEvent)
            self._sync_task = asyncio.create_task(
                self._client.sync_forever(
                    since=stored_token,
                    timeout=self._config.matrix_sync_timeout_ms,
                )
            )
        else:
            # Fresh start (or retry after interrupted backfill or first key import):
            # anchor position, backfill history, then begin live sync.  The sentinel
            # is written only after both phases complete, so an interruption here is
            # safe to retry.
            if stored_token:
                if not key_just_imported:
                    logger.warning(
                        "Stored sync token found but backfill sentinel absent — "
                        "previous backfill was interrupted; re-running backfill"
                    )

                # nio.AsyncClient.sync() falls back to self.loaded_sync_token when
                # no explicit `since` is given (async_client.py:1220).  On a retry
                # that would produce an incremental sync from the crash token, so
                # the returned prev_batch is newer than the original anchor.  With a
                # capped backfill_pages_max, post-crash traffic can then exhaust all
                # pages before reaching pre-crash history.  Clearing it forces a
                # genuinely tokenless sync and a prev_batch at the true room head.
                self._client.loaded_sync_token = ""

            logger.info("Performing initial sync to anchor timeline position")
            initial_sync: SyncResponse = await self._client.sync(full_state=True)

            await self._populate_rooms_from_sync(initial_sync)

            logger.info("Backfilling room history")
            await self._backfill(initial_sync)

            logger.info("Indexing initial sync timeline events")
            await self._index_initial_sync(initial_sync)

            self._mark_backfill_complete()
            await self._retry_pending_index()

            self._client.add_event_callback(self._on_message, RoomMessageText)
            self._client.add_event_callback(self._on_room_name, RoomNameEvent)
            self._client.add_event_callback(self._on_room_name, RoomAliasEvent)
            self._client.add_event_callback(self._on_room_member, RoomMemberEvent)

            logger.info("Starting live sync from token %s", initial_sync.next_batch)
            self._sync_task = asyncio.create_task(
                self._client.sync_forever(
                    since=initial_sync.next_batch,
                    timeout=self._config.matrix_sync_timeout_ms,
                )
            )

    async def stop(self) -> None:
        # Ask sync_forever to exit after the current iteration so _handle_sync
        # (token advance + callbacks) runs to completion before we snapshot state.
        # Hard-cancel only if the graceful exit hasn't happened within the poll
        # window; asyncio.shield keeps the task alive if wait_for times out first.
        if self._client:
            self._client.stop_sync_forever()
        if self._sync_task:
            try:
                await asyncio.wait_for(asyncio.shield(self._sync_task), timeout=35.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass
        self._store.close()
        if self._client:
            await self._client.close()

    async def get_recent_messages(
        self,
        k: int = 20,
        sender: Optional[str] = None,
        room_id: Optional[str] = None,
    ) -> list[MessageRecord]:
        return self._store.get_recent_messages(k, sender=sender, room_id=room_id)

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

        # events_before: reverse-chronological (closest to pivot first)
        # events_after: chronological (closest to pivot first)
        events_before = list(response.events_before or [])[:before]
        events_after = list(response.events_after or [])[:after]

        # Top up the before side by paginating backwards from the context start token.
        needed = before - len(events_before)
        if needed > 0:
            start_token = getattr(response, "start", None)
            if start_token:
                extra = await self._paginate_for_context(
                    room_id, start_token, nio.MessageDirection.back, needed
                )
                events_before.extend(extra)

        # Top up the after side by paginating forwards from the context end token.
        needed = after - len(events_after)
        if needed > 0:
            end_token = getattr(response, "end", None)
            if end_token:
                extra = await self._paginate_for_context(
                    room_id, end_token, nio.MessageDirection.front, needed
                )
                events_after.extend(extra)

        return {
            "event": self._event_to_dict(room_id, response.event) if response.event else None,
            "before": [self._event_to_dict(room_id, e) for e in events_before],
            "after": [self._event_to_dict(room_id, e) for e in events_after],
        }

    async def _paginate_for_context(
        self,
        room_id: str,
        token: str,
        direction: nio.MessageDirection,
        limit: int,
    ) -> list:
        response: RoomMessagesResponse = await self._client.room_messages(
            room_id=room_id,
            start=token,
            direction=direction,
            limit=limit,
        )
        if isinstance(response, nio.ErrorResponse):
            return []
        return list(response.chunk)[:limit]

    def get_room_info(self, room_id: str) -> dict:
        if self._client is None:
            raise RuntimeError("Client not started")
        info = self._store.get_room_info(room_id)
        if info is None:
            return {"error": f"Room {room_id} not found"}
        return info

    # -------------------------------------------------------------------------
    # Internal — startup helpers
    # -------------------------------------------------------------------------

    def _is_backfill_complete(self) -> bool:
        return self._store.get_meta(_META_BACKFILL_COMPLETE) is not None

    def _mark_backfill_complete(self) -> None:
        self._store.set_meta(_META_BACKFILL_COMPLETE, "1")

    async def _populate_rooms_from_sync(self, initial_sync: SyncResponse) -> None:
        """Write rooms and members from the initial full-state sync into the DB."""
        if not hasattr(initial_sync, "rooms") or not initial_sync.rooms:
            return
        joined = getattr(initial_sync.rooms, "join", {})
        for room_id in joined:
            room = self._client.rooms.get(room_id)
            if room is None:
                continue
            self._store.upsert_room(room_id, room.display_name or room_id, room.encrypted)
            for mxid, member in room.users.items():
                display_name = (
                    (member.display_name if member and member.display_name else None)
                    or self._sender_display_name(mxid)
                )
                self._store.upsert_member(room_id, mxid, display_name)

    def _restore_rooms_to_client(self) -> None:
        """Pre-populate nio's room dict from the DB so send_message works immediately
        on restart, before the first sync response arrives."""
        for room_data in self._store.get_all_rooms():
            room_id = room_data["room_id"]
            if room_id not in self._client.rooms:
                matrix_room = MatrixRoom(
                    room_id=room_id,
                    own_user_id=self._config.matrix_user_id,
                    encrypted=bool(room_data["encrypted"]),
                )
                self._client.rooms[room_id] = matrix_room

    # -------------------------------------------------------------------------
    # Internal — live sync callbacks
    # -------------------------------------------------------------------------

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        if room.room_id in self._config.ignored_room_ids:
            return
        member = room.users.get(event.sender)
        sender_name = (
            (member.display_name if member and member.display_name else None)
            or self._sender_display_name(event.sender)
        )
        record = MessageRecord(
            event_id=event.event_id,
            room_id=room.room_id,
            room_name=room.display_name or room.room_id,
            sender=event.sender,
            sender_name=sender_name,
            body=event.body,
            timestamp=event.server_timestamp,
        )
        # Persist before indexing; if the DB write fails we skip indexing entirely
        # so the event isn't silently lost — it will arrive again via live sync.
        try:
            inserted = self._store.insert_message(record, indexed=False)
        except Exception:
            logger.error(
                "Failed to persist message %s to DB; skipping indexing", record.event_id
            )
            return
        if not inserted:
            return  # duplicate event
        try:
            await self._index_message(record)
            self._store.mark_indexed(record.event_id)
        except Exception:
            logger.exception(
                "Failed to index message %s; will retry on next startup", record.event_id
            )

    async def _on_room_name(self, room: MatrixRoom, event) -> None:
        """Update the room's display name in the DB after a name/alias change."""
        self._store.update_room_name(room.room_id, room.display_name or room.room_id)

    async def _on_room_member(self, room: MatrixRoom, event: RoomMemberEvent) -> None:
        """Keep member list current on join/leave/display-name changes."""
        mxid = event.state_key  # user whose membership changed
        membership = event.membership
        if membership == "join":
            display_name = (
                getattr(event, "display_name", None) or self._sender_display_name(mxid)
            )
            # Ensure the room row exists (it may not if we just joined a new room)
            self._store.upsert_room(room.room_id, room.display_name or room.room_id, room.encrypted)
            self._store.upsert_member(room.room_id, mxid, display_name)
        elif membership in ("leave", "ban"):
            self._store.remove_member(room.room_id, mxid)

    # -------------------------------------------------------------------------
    # Internal — indexing
    # -------------------------------------------------------------------------

    async def _retry_pending_index(self) -> None:
        pending = self._store.get_pending_messages()
        if not pending:
            return
        logger.info("Re-indexing %d messages from pending index", len(pending))
        for record in pending:
            try:
                await self._index_message(record)
                self._store.mark_indexed(record.event_id)
            except Exception:
                logger.exception("Failed to re-index pending message %s", record.event_id)

    async def _index_message(self, record: MessageRecord) -> None:
        vector = await self._embedding_client.embed(record.body)
        await self._vector_store.upsert(record, vector)
        await self._webhook_dispatcher.dispatch(record)

    async def _batch_index(self, records: list[MessageRecord]) -> None:
        if not records:
            return
        # Insert into DB first; INSERT OR IGNORE deduplicates against previous runs.
        new_records = self._store.insert_messages_batch(records, indexed=False)
        if not new_records:
            return
        try:
            texts = [r.body for r in new_records]
            vectors = await self._embedding_client.embed_batch(texts)
            for record, vector in zip(new_records, vectors):
                await self._vector_store.upsert(record, vector)
            self._store.mark_indexed_batch([r.event_id for r in new_records])
        except Exception:
            logger.exception("Batch indexing failed for %d records", len(new_records))
            # Records remain in DB with indexed=0 and will be retried on next startup.

    # -------------------------------------------------------------------------
    # Internal — backfill
    # -------------------------------------------------------------------------

    async def _index_initial_sync(self, initial_sync: SyncResponse) -> None:
        if not hasattr(initial_sync, "rooms") or not initial_sync.rooms:
            return
        joined = getattr(initial_sync.rooms, "join", {})
        ignored = self._config.ignored_room_ids
        records: list[MessageRecord] = []
        for room_id, room_info in joined.items():
            if room_id in ignored:
                continue
            timeline = getattr(room_info, "timeline", None)
            if timeline is None:
                continue
            for event in getattr(timeline, "events", []) or []:
                if isinstance(event, RoomMessageText):
                    record = self._parse_event(room_id, event)
                    if record:
                        records.append(record)
        if records:
            await self._batch_index(records)
        logger.info("Indexed %d messages from initial sync timeline", len(records))

    async def _backfill(self, initial_sync: SyncResponse) -> None:
        rooms_response: JoinedRoomsResponse = await self._client.joined_rooms()
        ignored = self._config.ignored_room_ids
        room_ids = [
            r for r in (rooms_response.rooms if hasattr(rooms_response, "rooms") else [])
            if r not in ignored
        ]

        for room_id in room_ids:
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

            await self._backfill_room(room_id, prev_batch)

    async def _backfill_room(self, room_id: str, prev_batch: str) -> None:
        pages_max = self._config.backfill_pages_max  # 0 = unlimited
        page = 0
        fetched = 0

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

            page_records = [
                record
                for event in response.chunk
                if isinstance(event, RoomMessageText)
                for record in [self._parse_event(room_id, event)]
                if record is not None
            ]
            if page_records:
                await self._batch_index(page_records)

            fetched += len(response.chunk)
            page += 1

            # Spec-correct termination: end absent means no more pages
            if response.end is None:
                break
            prev_batch = response.end

        logger.info("Backfilled %d messages from room %s (%d pages)", fetched, room_id, page)

    # -------------------------------------------------------------------------
    # Internal — key backup
    # -------------------------------------------------------------------------

    def _compute_key_backup_fingerprint(self) -> str:
        h = hashlib.sha256()
        with open(self._config.matrix_key_backup_file, "rb") as f:
            h.update(f.read())
        h.update(self._config.matrix_key_backup_passphrase.encode())
        return h.hexdigest()

    async def _import_key_backup(self) -> bool:
        """Import the E2EE key backup if configured and not already imported.

        Returns True if the import actually ran this call, False if skipped.
        Raises on misconfiguration (missing file, changed key/passphrase).
        """
        if not self._config.matrix_key_backup_file:
            return False
        backup_file = self._config.matrix_key_backup_file
        if not os.path.exists(backup_file):
            raise FileNotFoundError(
                f"MATRIX_KEY_BACKUP_FILE is set to {backup_file!r} but the file does not exist"
            )
        stored = self._store.get_meta(_META_KEY_BACKUP)
        if stored is not None:
            current = self._compute_key_backup_fingerprint()
            if stored != current:
                raise RuntimeError(
                    f"MATRIX_KEY_BACKUP_FILE or MATRIX_KEY_BACKUP_PASSPHRASE has changed "
                    f"since the last import (or the sentinel predates fingerprint tracking). "
                    f"Delete the '{_META_KEY_BACKUP}' row from the meta table to re-import."
                )
            logger.debug("E2EE key backup already imported; skipping")
            return False
        logger.info("Importing E2EE key backup from %s", backup_file)
        await self._client.import_keys(backup_file, self._config.matrix_key_backup_passphrase)
        try:
            self._store.set_meta(_META_KEY_BACKUP, self._compute_key_backup_fingerprint())
        except Exception:
            logger.exception(
                "Failed to write key backup sentinel; import will run again on next start"
            )
        logger.info("E2EE key backup imported successfully")
        return True

    # -------------------------------------------------------------------------
    # Internal — helpers
    # -------------------------------------------------------------------------

    def _sender_display_name(self, sender: str) -> str:
        if sender.startswith("@") and ":" in sender:
            return sender[1:sender.index(":")]
        return sender

    def _resolve_room_name(self, room_id: str) -> str:
        if self._client:
            room = self._client.rooms.get(room_id)
            if room and room.display_name:
                return room.display_name
        name = self._store.get_room_display_name(room_id)
        return name or room_id

    def _resolve_display_name(self, room_id: str, sender: str) -> str:
        if self._client:
            room = self._client.rooms.get(room_id)
            if room:
                member = room.users.get(sender)
                if member and member.display_name:
                    return member.display_name
        return self._sender_display_name(sender)

    def _parse_event(self, room_id: str, event: RoomMessageText) -> Optional[MessageRecord]:
        body = getattr(event, "body", None)
        if not body:
            return None
        return MessageRecord(
            event_id=event.event_id,
            room_id=room_id,
            room_name=self._resolve_room_name(room_id),
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
