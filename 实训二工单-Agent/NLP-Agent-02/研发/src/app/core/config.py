from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="schedule-reminder-agent", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    app_role: str = Field(default="api", alias="APP_ROLE")
    database_url: str = Field(default="sqlite+aiosqlite:///./app.db", alias="DATABASE_URL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    timezone: str = Field(default="Asia/Shanghai", alias="TIMEZONE")
    reminder_channel: str = Field(default="console", alias="REMINDER_CHANNEL")
    reminder_webhook_url: str | None = Field(default=None, alias="REMINDER_WEBHOOK_URL")
    reminder_max_attempts: int = Field(default=3, alias="REMINDER_MAX_ATTEMPTS")
    reminder_retry_delay_seconds: int = Field(default=60, alias="REMINDER_RETRY_DELAY_SECONDS")
    reminder_retry_backoff_multiplier: float = Field(default=2.0, alias="REMINDER_RETRY_BACKOFF_MULTIPLIER")
    reminder_retry_max_delay_seconds: int = Field(default=1800, alias="REMINDER_RETRY_MAX_DELAY_SECONDS")
    reminder_alert_channel: str = Field(default="console", alias="REMINDER_ALERT_CHANNEL")
    reminder_alert_webhook_url: str | None = Field(default=None, alias="REMINDER_ALERT_WEBHOOK_URL")
    scheduler_lock_enabled: bool = Field(default=True, alias="SCHEDULER_LOCK_ENABLED")
    scheduler_lock_owner: str = Field(default="local-scheduler", alias="SCHEDULER_LOCK_OWNER")
    scheduler_lock_ttl_seconds: int = Field(default=120, alias="SCHEDULER_LOCK_TTL_SECONDS")
    worker_owner: str = Field(default="local-worker", alias="WORKER_OWNER")
    worker_poll_interval_seconds: int = Field(default=10, alias="WORKER_POLL_INTERVAL_SECONDS")
    worker_batch_size: int = Field(default=20, alias="WORKER_BATCH_SIZE")
    worker_lock_timeout_seconds: int = Field(default=300, alias="WORKER_LOCK_TIMEOUT_SECONDS")
    redis_enabled: bool = Field(default=False, alias="REDIS_ENABLED")
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", alias="REDIS_URL")
    redis_queue_key: str = Field(default="schedule:delivery:queue", alias="REDIS_QUEUE_KEY")
    auto_create_tables: bool = Field(default=False, alias="AUTO_CREATE_TABLES")
    llm_enabled: bool = Field(default=False, alias="LLM_ENABLED")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=400, alias="LLM_MAX_TOKENS")
    llm_debug_logging: bool = Field(default=False, alias="LLM_DEBUG_LOGGING")
    llm_reply_enabled: bool = Field(default=False, alias="LLM_REPLY_ENABLED")
    admin_access_token: str | None = Field(default=None, alias="ADMIN_ACCESS_TOKEN")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
