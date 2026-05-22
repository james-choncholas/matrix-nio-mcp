from functools import lru_cache
from pathlib import Path
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

    # OpenAI
    openai_api_key: str

    # Webhook
    webhook_url: str = ""
    webhook_secret: str = ""

    # Behaviour
    backfill_limit: int = 100
    backfill_pages_max: int = 10  # 0 = unlimited
    message_buffer_size: int = 500
    matrix_sync_timeout_ms: int = 30000
    sse_queue_maxsize: int = 100
    sse_port: int = 8000
    allow_send_message: bool = False  # set ALLOW_SEND_MESSAGE=true to enable


@lru_cache
def get_settings() -> Settings:
    return Settings()
