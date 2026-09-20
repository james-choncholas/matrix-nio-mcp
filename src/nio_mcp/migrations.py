"""Retroactive + self-healing reconciliation of message sender display names.

A message is indexed with the author's Matrix display name, falling back to the
MXID localpart when the name is not yet known (common for bridge ghost users whose
profile propagates asynchronously). That fallback was historically permanent: once
the real name arrived, nothing revisited the already-indexed message.

`reconcile_sender_names` (run at startup and available as a standalone script) and
`apply_sender_name` (called live from the member-event callback) close that gap by
rewriting the stored name in both SQLite and Qdrant. No re-embedding is needed —
the vector derives from the message body only.
"""

import logging

from nio_mcp.store import MessageStore
from nio_mcp.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _localpart(sender: str) -> str:
    """Mirror of matrix_client._sender_display_name — the fallback name shape."""
    if sender.startswith("@") and ":" in sender:
        return sender[1:sender.index(":")]
    return sender


async def apply_sender_name(
    store: MessageStore,
    vector_store: VectorStore,
    room_id: str,
    sender: str,
    display_name: str,
) -> int:
    """Propagate a known display name to a sender's already-stored messages.

    Updates Qdrant before committing the corresponding SQLite change. If Qdrant is
    unavailable, SQLite remains stale so a later reconciliation can retry safely.
    Returns the number of SQLite messages updated.
    """
    if not store.count_sender_name_updates(room_id, sender, display_name):
        return 0

    await vector_store.update_sender_name(room_id, sender, display_name)
    return store.update_sender_name(room_id, sender, display_name)


async def reconcile_sender_names(
    store: MessageStore, vector_store: VectorStore
) -> int:
    """Fix every message whose stored sender_name lags the known member name.

    Idempotent: after a successful pass the SQLite/Qdrant names match the roster, so
    subsequent runs find nothing and issue no writes. Returns the count of senders
    whose messages were updated. Fallback (localpart) member names are skipped so a
    real, already-stored name is never downgraded.
    """
    fixed = 0
    for room_id, sender, display_name in store.get_sender_name_fixes():
        if not display_name or display_name == _localpart(sender):
            continue
        try:
            if await apply_sender_name(store, vector_store, room_id, sender, display_name):
                fixed += 1
        except Exception:
            logger.exception(
                "Failed to reconcile sender_name for %s in %s", sender, room_id
            )
    if fixed:
        logger.info("Reconciled sender_name for %d sender(s)", fixed)
    return fixed
