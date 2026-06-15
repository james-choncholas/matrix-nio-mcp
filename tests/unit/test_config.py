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
    assert s.webhook_bearer_token == ""
    assert s.webhook_prompt_header == "New Matrix messages:"
    assert s.webhook_prompt_per_msg == "{sender_name} ({sender}) in {room_name} ({room}): {message}"
    assert s.webhook_model == "gpt-4o-mini"
    assert s.webhook_cooldown_seconds == 300.0
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
        "WEBHOOK_URL": "http://llm.example.com/v1",
        "WEBHOOK_BEARER_TOKEN": "secret-token",
        "WEBHOOK_PROMPT_HEADER": "Summarize these:",
        "WEBHOOK_PROMPT_PER_MSG": "{sender_name}: {message}",
        "WEBHOOK_MODEL": "gpt-4.1-mini",
        "WEBHOOK_COOLDOWN_SECONDS": "12.5",
        "BACKFILL_LIMIT": "50",
        "MCP_PORT": "9000",
    })
    assert s.qdrant_host == "qdrant.internal"
    assert s.qdrant_port == 6334
    assert s.webhook_url == "http://llm.example.com/v1"
    assert s.webhook_bearer_token == "secret-token"
    assert s.webhook_prompt_header == "Summarize these:"
    assert s.webhook_prompt_per_msg == "{sender_name}: {message}"
    assert s.webhook_model == "gpt-4.1-mini"
    assert s.webhook_cooldown_seconds == 12.5
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


@pytest.mark.parametrize("value", ["0", "-1", "-0.5"])
def test_invalid_webhook_cooldown_seconds_raise_validation_error(monkeypatch, value):
    for key, val in REQUIRED_ENV.items():
        monkeypatch.setenv(key, val)
    monkeypatch.setenv("WEBHOOK_COOLDOWN_SECONDS", value)
    with pytest.raises(ValidationError):
        Settings()


# --- key backup paired-field validator ---

def test_key_backup_file_without_passphrase_raises(monkeypatch):
    with pytest.raises(ValidationError, match="MATRIX_KEY_BACKUP_FILE and MATRIX_KEY_BACKUP_PASSPHRASE"):
        _make_settings(monkeypatch, {"MATRIX_KEY_BACKUP_FILE": "/tmp/keys.txt"})


def test_key_backup_passphrase_without_file_raises(monkeypatch):
    with pytest.raises(ValidationError, match="MATRIX_KEY_BACKUP_FILE and MATRIX_KEY_BACKUP_PASSPHRASE"):
        _make_settings(monkeypatch, {"MATRIX_KEY_BACKUP_PASSPHRASE": "secret"})


def test_key_backup_both_set_is_valid(monkeypatch):
    s = _make_settings(monkeypatch, {
        "MATRIX_KEY_BACKUP_FILE": "/tmp/keys.txt",
        "MATRIX_KEY_BACKUP_PASSPHRASE": "secret",
    })
    assert s.matrix_key_backup_file == "/tmp/keys.txt"
    assert s.matrix_key_backup_passphrase == "secret"


def test_key_backup_neither_set_is_valid(monkeypatch):
    s = _make_settings(monkeypatch)
    assert s.matrix_key_backup_file == ""
    assert s.matrix_key_backup_passphrase == ""


# --- IGNORED_ROOMS ---

def test_ignored_rooms_default_is_empty(monkeypatch):
    s = _make_settings(monkeypatch)
    assert s.ignored_room_ids == frozenset()


def test_ignored_rooms_unset_env_is_empty(monkeypatch):
    monkeypatch.delenv("IGNORED_ROOMS", raising=False)
    s = _make_settings(monkeypatch)
    assert s.ignored_room_ids == frozenset()


def test_ignored_rooms_empty_string_is_empty(monkeypatch):
    s = _make_settings(monkeypatch, {"IGNORED_ROOMS": ""})
    assert s.ignored_room_ids == frozenset()


def test_ignored_rooms_single_entry(monkeypatch):
    s = _make_settings(monkeypatch, {"IGNORED_ROOMS": "!abc:example.org"})
    assert s.ignored_room_ids == frozenset({"!abc:example.org"})


def test_ignored_rooms_comma_separated(monkeypatch):
    s = _make_settings(monkeypatch, {"IGNORED_ROOMS": "!abc:example.org,!def:example.org"})
    assert s.ignored_room_ids == frozenset({"!abc:example.org", "!def:example.org"})


def test_ignored_rooms_trims_whitespace(monkeypatch):
    s = _make_settings(monkeypatch, {"IGNORED_ROOMS": " !abc:example.org , !def:example.org "})
    assert s.ignored_room_ids == frozenset({"!abc:example.org", "!def:example.org"})
