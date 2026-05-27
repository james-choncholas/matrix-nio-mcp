import pytest
from pydantic import ValidationError

from nio_mcp.config import Settings, get_settings

REQUIRED_ENV = {
    "MATRIX_HOMESERVER_URL": "https://matrix.example.org",
    "MATRIX_ACCESS_TOKEN": "syt_token",
    "MATRIX_USER_ID": "@bot:example.org",
    "MATRIX_DEVICE_ID": "DEVID123",
    "OPENAI_API_KEY": "sk-test",
}


def _make_settings(monkeypatch, overrides=None):
    for key, val in REQUIRED_ENV.items():
        monkeypatch.setenv(key, val)
    if overrides:
        for key, val in overrides.items():
            monkeypatch.setenv(key, val)
    return Settings()


def test_required_fields_raise_without_env(monkeypatch):
    for key in REQUIRED_ENV:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_default_values(monkeypatch):
    from pathlib import Path
    s = _make_settings(monkeypatch)
    assert s.matrix_store_path == str(Path.home() / ".cache" / "nio-mcp" / "store")
    assert s.qdrant_host == "localhost"
    assert s.qdrant_port == 6333
    assert s.qdrant_collection == "matrix_messages"
    assert s.webhook_url == ""
    assert s.webhook_secret == ""
    assert s.backfill_limit == 100
    assert s.backfill_pages_max == 10
    assert s.message_buffer_size == 500
    assert s.matrix_sync_timeout_ms == 30000
    assert s.sse_queue_maxsize == 100
    assert s.mcp_port == 8000


def test_env_overrides_defaults(monkeypatch):
    s = _make_settings(monkeypatch, {
        "QDRANT_HOST": "qdrant.internal",
        "QDRANT_PORT": "6334",
        "BACKFILL_LIMIT": "50",
        "MCP_PORT": "9000",
    })
    assert s.qdrant_host == "qdrant.internal"
    assert s.qdrant_port == 6334
    assert s.backfill_limit == 50
    assert s.mcp_port == 9000


def test_required_fields_loaded_from_env(monkeypatch):
    s = _make_settings(monkeypatch)
    assert s.matrix_homeserver_url == "https://matrix.example.org"
    assert s.matrix_access_token == "syt_token"
    assert s.matrix_user_id == "@bot:example.org"
    assert s.matrix_device_id == "DEVID123"
    assert s.openai_api_key == "sk-test"


def test_get_settings_returns_settings_instance(monkeypatch):
    for key, val in REQUIRED_ENV.items():
        monkeypatch.setenv(key, val)
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert isinstance(s, Settings)
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("field,value", [
    ("SSE_QUEUE_MAXSIZE", "0"),
    ("SSE_QUEUE_MAXSIZE", "-1"),
    ("BACKFILL_LIMIT", "0"),
    ("MESSAGE_BUFFER_SIZE", "-5"),
    ("MATRIX_SYNC_TIMEOUT_MS", "0"),
    ("MCP_SESSION_TIMEOUT", "-1"),
    ("MCP_PORT", "0"),
    ("MCP_PORT", "99999"),
    ("QDRANT_PORT", "65536"),
    ("BACKFILL_PAGES_MAX", "-1"),
])
def test_invalid_numeric_fields_raise_validation_error(monkeypatch, field, value):
    for key, val in REQUIRED_ENV.items():
        monkeypatch.setenv(key, val)
    monkeypatch.setenv(field, value)
    with pytest.raises(ValidationError):
        Settings()


def test_backfill_pages_max_zero_is_valid(monkeypatch):
    s = _make_settings(monkeypatch, {"BACKFILL_PAGES_MAX": "0"})
    assert s.backfill_pages_max == 0


# --- key backup paired-field validator ---

def test_key_backup_content_without_passphrase_raises(monkeypatch):
    with pytest.raises(ValidationError, match="MATRIX_KEY_BACKUP_CONTENT and MATRIX_KEY_BACKUP_PASSPHRASE"):
        _make_settings(monkeypatch, {"MATRIX_KEY_BACKUP_CONTENT": "-----BEGIN MEGOLM SESSION DATA-----\nABC\n-----END MEGOLM SESSION DATA-----"})


def test_key_backup_passphrase_without_content_raises(monkeypatch):
    with pytest.raises(ValidationError, match="MATRIX_KEY_BACKUP_CONTENT and MATRIX_KEY_BACKUP_PASSPHRASE"):
        _make_settings(monkeypatch, {"MATRIX_KEY_BACKUP_PASSPHRASE": "secret"})


def test_key_backup_both_set_is_valid(monkeypatch):
    s = _make_settings(monkeypatch, {
        "MATRIX_KEY_BACKUP_CONTENT": "-----BEGIN MEGOLM SESSION DATA-----\nABC\n-----END MEGOLM SESSION DATA-----",
        "MATRIX_KEY_BACKUP_PASSPHRASE": "secret",
    })
    assert s.matrix_key_backup_passphrase == "secret"


def test_key_backup_neither_set_is_valid(monkeypatch):
    s = _make_settings(monkeypatch)
    assert s.matrix_key_backup_content == ""
    assert s.matrix_key_backup_passphrase == ""
