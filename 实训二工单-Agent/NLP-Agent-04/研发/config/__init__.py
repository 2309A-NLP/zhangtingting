"""全局配置 — Pydantic Settings"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple, Type

from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # 只读 .env 文件，不读系统环境变量（避免旧变量干扰）
        return (init_settings, dotenv_settings)

    # ── 数据库 ──
    DB_PATH: str = "data/raw/bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db"

    # ── LLM ──
    LLM_PROVIDER: Literal["openai", "deepseek", "dashscope", "local"] = "openai"
    LLM_API_KEY: str = ""
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT: int = 20
    LLM_MODEL_PATH: str = ""  # 本地模型路径

    # ── Embedding ──
    EMBEDDING_MODEL_NAME: str = "shibing624/text2vec-base-chinese"

    # ── 向量数据库 ──
    VECTOR_DB_PATH: str = "data/chroma_db"
    VECTOR_DB_COLLECTION: str = "few_shot_sql"

    # ── 缓存 ──
    CACHE_TYPE: Literal["memory", "redis"] = "memory"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Few-shot ──
    FEW_SHOT_PATH: str = "data/processed/few_shot_examples.json"
    FEW_SHOT_TOP_K: int = 2

    # ── API ──
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 2
    API_CORS_ORIGINS: str = "*"

    # ── 认证 ──
    API_KEYS: str = ""  # 逗号分隔
    API_KEY_ENABLED: bool = True

    # ── 日志 ──
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # ── 限流 ──
    RATE_LIMIT_PER_MIN: int = 100

    # ── SQL 执行 ──
    SQL_TIMEOUT_SECONDS: int = 60
    SQL_MAX_ROWS: int = 200
    SQL_MAX_RETRIES: int = 2

    @property
    def DB_PATH_RESOLVED(self) -> Path:
        return Path(self.DB_PATH)

    @property
    def VECTOR_DB_PATH_RESOLVED(self) -> Path:
        return Path(self.VECTOR_DB_PATH)

    @property
    def API_KEYS_LIST(self) -> list[str]:
        return [k.strip() for k in self.API_KEYS.split(",") if k.strip()]

    @property
    def CORS_ORIGINS_LIST(self) -> list[str]:
        origins = self.API_CORS_ORIGINS
        if origins == "*":
            return ["*"]
        return [o.strip() for o in origins.split(",") if o.strip()]

    @property
    def LLM_KWARGS(self) -> dict:
        return {
            "temperature": self.LLM_TEMPERATURE,
            "max_tokens": self.LLM_MAX_TOKENS,
            "timeout": self.LLM_TIMEOUT,
        }


settings = Settings()
