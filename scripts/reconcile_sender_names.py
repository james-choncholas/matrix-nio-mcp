#!/usr/bin/env python3
"""One-off / repeatable reconciliation of message sender display names.

Rewrites the `sender_name` (and derived `sender_search`) stored in SQLite and
Qdrant for any message that was indexed with the MXID-localpart fallback before its
author's real display name was known. Reads the current names from the member
roster in the local SQLite store; no Matrix connection is required.

This is the same routine `MatrixMCPClient.start()` runs on every startup, exposed as
a standalone entry point for manual runs:

    python scripts/reconcile_sender_names.py

It is idempotent — a second run with nothing to fix performs no writes.
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nio_mcp.config import Settings  # noqa: E402
from nio_mcp.migrations import reconcile_sender_names  # noqa: E402
from nio_mcp.store import MessageStore  # noqa: E402
from nio_mcp.vector_store import VectorStore  # noqa: E402


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings()

    store = MessageStore(os.path.join(settings.matrix_store_path, "nio_mcp.db"))
    store.open()
    vector_store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=settings.qdrant_collection,
    )
    try:
        fixed = await reconcile_sender_names(store, vector_store)
    finally:
        store.close()
        await vector_store.close()

    logging.info("Done. Reconciled sender_name for %d sender(s).", fixed)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
