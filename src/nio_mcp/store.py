import sqlite3
import logging
from typing import Optional

from nio_mcp.models import MessageRecord

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rooms (
    room_id      TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    encrypted    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS members (
    room_id      TEXT NOT NULL,
    mxid         TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (room_id, mxid)
);
CREATE TABLE IF NOT EXISTS messages (
    event_id    TEXT PRIMARY KEY,
    room_id     TEXT NOT NULL,
    room_name   TEXT NOT NULL,
    sender      TEXT NOT NULL,
    sender_name TEXT NOT NULL,
    body        TEXT NOT NULL,
    timestamp   INTEGER NOT NULL,
    indexed     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_ts      ON messages (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_messages_room    ON messages (room_id);
CREATE INDEX IF NOT EXISTS idx_messages_sender  ON messages (sender);
CREATE INDEX IF NOT EXISTS idx_messages_pending ON messages (indexed) WHERE indexed = 0;
"""


class MessageStore:
    """SQLite-backed store for rooms, members, messages, and app metadata."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def open(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Meta / sentinels
    # ------------------------------------------------------------------

    def get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def delete_meta(self, key: str) -> None:
        self._conn.execute("DELETE FROM meta WHERE key = ?", (key,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Rooms
    # ------------------------------------------------------------------

    def upsert_room(self, room_id: str, display_name: str, encrypted: bool = False) -> None:
        self._conn.execute(
            """INSERT INTO rooms (room_id, display_name, encrypted)
               VALUES (?, ?, ?)
               ON CONFLICT(room_id) DO UPDATE SET
                   display_name = excluded.display_name,
                   encrypted    = excluded.encrypted""",
            (room_id, display_name, int(encrypted)),
        )
        self._conn.commit()

    def update_room_name(self, room_id: str, display_name: str) -> None:
        self._conn.execute(
            "UPDATE rooms SET display_name = ? WHERE room_id = ?",
            (display_name, room_id),
        )
        self._conn.commit()

    def get_room_display_name(self, room_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT display_name FROM rooms WHERE room_id = ?", (room_id,)
        ).fetchone()
        return row["display_name"] if row else None

    def get_all_rooms(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT room_id, display_name, encrypted FROM rooms"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_room_info(self, room_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT display_name FROM rooms WHERE room_id = ?", (room_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "room_id": room_id,
            "name": row["display_name"],
            "members": self.get_members(room_id),
        }

    # ------------------------------------------------------------------
    # Members
    # ------------------------------------------------------------------

    def upsert_member(self, room_id: str, mxid: str, display_name: str) -> None:
        self._conn.execute(
            """INSERT INTO members (room_id, mxid, display_name)
               VALUES (?, ?, ?)
               ON CONFLICT(room_id, mxid) DO UPDATE SET display_name = excluded.display_name""",
            (room_id, mxid, display_name),
        )
        self._conn.commit()

    def remove_member(self, room_id: str, mxid: str) -> None:
        self._conn.execute(
            "DELETE FROM members WHERE room_id = ? AND mxid = ?", (room_id, mxid)
        )
        self._conn.commit()

    def get_members(self, room_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT mxid, display_name FROM members WHERE room_id = ?", (room_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def insert_message(self, record: MessageRecord, indexed: bool = False) -> bool:
        """Insert a message; returns True if newly inserted, False if duplicate."""
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO messages
               (event_id, room_id, room_name, sender, sender_name, body, timestamp, indexed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.event_id, record.room_id, record.room_name,
                record.sender, record.sender_name, record.body,
                record.timestamp, int(indexed),
            ),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def insert_messages_batch(
        self, records: list[MessageRecord], indexed: bool = False
    ) -> list[MessageRecord]:
        """Insert many messages; return those newly inserted (duplicates skipped)."""
        new_records = []
        for record in records:
            cur = self._conn.execute(
                """INSERT OR IGNORE INTO messages
                   (event_id, room_id, room_name, sender, sender_name, body, timestamp, indexed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.event_id, record.room_id, record.room_name,
                    record.sender, record.sender_name, record.body,
                    record.timestamp, int(indexed),
                ),
            )
            if cur.rowcount > 0:
                new_records.append(record)
        self._conn.commit()
        return new_records

    def mark_indexed(self, event_id: str) -> None:
        self._conn.execute(
            "UPDATE messages SET indexed = 1 WHERE event_id = ?", (event_id,)
        )
        self._conn.commit()

    def mark_indexed_batch(self, event_ids: list[str]) -> None:
        self._conn.executemany(
            "UPDATE messages SET indexed = 1 WHERE event_id = ?",
            [(eid,) for eid in event_ids],
        )
        self._conn.commit()

    def get_pending_messages(self) -> list[MessageRecord]:
        rows = self._conn.execute(
            """SELECT event_id, room_id, room_name, sender, sender_name, body, timestamp
               FROM messages WHERE indexed = 0 ORDER BY timestamp ASC"""
        ).fetchall()
        return [MessageRecord(**dict(r)) for r in rows]

    def get_recent_messages(
        self,
        k: int,
        sender: Optional[str] = None,
        room_id: Optional[str] = None,
    ) -> list[MessageRecord]:
        conditions: list[str] = []
        params: list = []
        if sender:
            conditions.append("sender = ?")
            params.append(sender)
        if room_id:
            conditions.append("room_id = ?")
            params.append(room_id)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(k)
        rows = self._conn.execute(
            f"""SELECT event_id, room_id, room_name, sender, sender_name, body, timestamp
                FROM messages {where}
                ORDER BY timestamp DESC LIMIT ?""",
            params,
        ).fetchall()
        # Reverse so result is oldest-first (consistent with previous behaviour)
        return [MessageRecord(**dict(r)) for r in reversed(rows)]
