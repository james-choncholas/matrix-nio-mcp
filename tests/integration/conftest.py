import os
import pytest
import asyncio


QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as requiring external services"
    )


def qdrant_is_reachable() -> bool:
    import socket
    try:
        with socket.create_connection((QDRANT_HOST, QDRANT_PORT), timeout=2):
            return True
    except OSError:
        return False


skip_if_no_qdrant = pytest.mark.skipif(
    not qdrant_is_reachable(),
    reason=f"Qdrant not reachable at {QDRANT_HOST}:{QDRANT_PORT}",
)
