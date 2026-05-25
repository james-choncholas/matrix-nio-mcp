from functools import lru_cache
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Matrix — all required
    matrix_homeserver_url: str
    matrix_access_token: str
    matrix_user_id: str
    matrix_device_id: str  # required: E2EE store needs a stable device identity
    matrix_store_path: str = str(Path.home() / ".cache" / "nio-mcp" / "store")

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "matrix_messages"

    # OpenAI / Embeddings
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_vector_size: int = 1536

    # Webhook
    webhook_url: str = ""
    webhook_secret: str = ""

    # Behaviour
    backfill_limit: int = 100
    backfill_pages_max: int = 10  # 0 = unlimited
    message_buffer_size: int = 500
    matrix_sync_timeout_ms: int = 30000
    sse_queue_maxsize: int = 100
    mcp_port: int = 8000
    mcp_session_timeout: int = 1800  # seconds; idle sessions are reaped after this long
    allow_send_message: bool = False  # set ALLOW_SEND_MESSAGE=true to enable

    @field_validator(
        "backfill_limit", "message_buffer_size", "matrix_sync_timeout_ms",
        "sse_queue_maxsize", "mcp_session_timeout", "embedding_vector_size",
    )
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
