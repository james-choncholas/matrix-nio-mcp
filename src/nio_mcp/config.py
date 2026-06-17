from functools import lru_cache
from pathlib import Path
from pydantic import computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Matrix — all required
    matrix_homeserver_url: str
    matrix_access_token: str
    matrix_user_id: str
    matrix_device_id: str  # required: E2EE store needs a stable device identity
    matrix_store_path: str = str(Path.home() / ".cache" / "nio-mcp" / "store")
    # Optional: path to an Element-exported E2EE key file and its export passphrase.
    # When set, session keys are imported into the Olm store on first run (idempotent).
    matrix_key_backup_file: str = ""
    matrix_key_backup_passphrase: str = ""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "matrix_messages"

    # OpenAI / Embeddings
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_vector_size: int = 1536
    embedding_max_tokens: int = 8191  # truncate texts longer than this before embedding

    # Webhook / LLM callback
    webhook_url: str = ""  # OpenAI-compatible base URL, e.g. https://api.openai.com/v1
    webhook_bearer_token: str = ""
    webhook_prompt_header: str = "New Matrix messages:"  # appears once before all message lines
    webhook_prompt_per_msg: str = "{sender_name} ({sender}) in {room_name} ({room}): {message}"
    webhook_model: str = "gpt-4o-mini"
    webhook_cooldown_seconds: float = 300.0  # fire LLM only after this many quiet seconds
    webhook_tools: str = ""  # JSON object merged into the chat completions body, e.g. '{"tool_ids": ["server:mcp:myserver"]}'

    # Behaviour
    backfill_limit: int = 100
    backfill_pages_max: int = 10  # 0 = unlimited
    message_buffer_size: int = 500
    matrix_sync_timeout_ms: int = 30000
    sse_queue_maxsize: int = 100
    mcp_port: int = 8000
    mcp_session_timeout: int = 1800  # seconds; idle sessions are reaped after this long
    allow_send_message: bool = False  # set ALLOW_SEND_MESSAGE=true to enable
    http_auth_token: str = ""  # if set, require Bearer token in Authorization header
    ignored_rooms: str = ""  # IGNORED_ROOMS env var: comma-separated room IDs to skip

    @computed_field  # type: ignore[misc]
    @property
    def ignored_room_ids(self) -> frozenset[str]:
        return frozenset(r.strip() for r in self.ignored_rooms.split(",") if r.strip())

    @field_validator(
        "backfill_limit", "message_buffer_size", "matrix_sync_timeout_ms",
        "sse_queue_maxsize", "mcp_session_timeout", "embedding_vector_size",
        "embedding_max_tokens",
    )
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("webhook_cooldown_seconds")
    @classmethod
    def cooldown_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be a positive number")
        return v

    @field_validator("backfill_pages_max")
    @classmethod
    def must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be 0 (unlimited) or a positive integer")
        return v

    @field_validator("mcp_port", "qdrant_port")
    @classmethod
    def valid_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("must be a valid port number (1–65535)")
        return v

    @model_validator(mode="after")
    def key_backup_fields_must_be_paired(self) -> "Settings":
        has_file = bool(self.matrix_key_backup_file)
        has_passphrase = bool(self.matrix_key_backup_passphrase)
        if has_file != has_passphrase:
            raise ValueError(
                "MATRIX_KEY_BACKUP_FILE and MATRIX_KEY_BACKUP_PASSPHRASE "
                "must be set together or not at all"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
