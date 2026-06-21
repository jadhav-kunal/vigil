"""Typed configuration for Vigil (pydantic-settings).

Every optional integration auto-detects its key here. Absent key => the capability is skipped
silently downstream (Invariant I2). Env var names are explicit (mixed conventions: OPENAI_*,
VIGIL_*, TTC_*, etc.), so each field declares its own validation alias.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Core proxy ---
    host: str = Field(default="0.0.0.0", validation_alias="VIGIL_HOST")
    port: int = Field(default=8765, validation_alias="VIGIL_PORT")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", validation_alias="OPENAI_BASE_URL"
    )
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com", validation_alias="ANTHROPIC_BASE_URL"
    )
    upstream_timeout_s: float = Field(default=120.0, validation_alias="VIGIL_UPSTREAM_TIMEOUT")
    log_level: str = Field(default="INFO", validation_alias="VIGIL_LOG_LEVEL")
    price_table_path: str | None = Field(default=None, validation_alias="VIGIL_PRICE_TABLE")

    # --- Store ---
    db_path: str = Field(default="vigil.db", validation_alias="VIGIL_DB_PATH")
    redis_url: str | None = Field(default=None, validation_alias="REDIS_URL")

    # --- Watchdog thresholds (spec 4.2) ---
    window: int = Field(default=5, validation_alias="VIGIL_WINDOW")
    trip_streak: int = Field(default=3, validation_alias="VIGIL_TRIP_STREAK")
    theta_sim: float = Field(default=0.85, validation_alias="VIGIL_THETA_SIM")
    theta_ent: float = Field(default=0.30, validation_alias="VIGIL_THETA_ENT")
    goal_judge_cadence: int = Field(default=5, validation_alias="VIGIL_GOAL_JUDGE_CADENCE")
    recovery_steps: int = Field(default=3, validation_alias="VIGIL_RECOVERY_STEPS")
    session_mode: str = Field(default="normal", validation_alias="VIGIL_SESSION_MODE")

    # --- Circuit breaker (spec 4.3) ---
    breaker_downgrade_openai: str = Field(
        default="gpt-4o-mini", validation_alias="VIGIL_BREAKER_DOWNGRADE_OPENAI"
    )
    breaker_downgrade_anthropic: str = Field(
        default="claude-3-5-haiku-latest", validation_alias="VIGIL_BREAKER_DOWNGRADE_ANTHROPIC"
    )
    # Horizon used to estimate the loop cost the breaker capped (for the dashboard meter).
    breaker_projection_steps: int = Field(
        default=50, validation_alias="VIGIL_BREAKER_PROJECTION_STEPS"
    )

    # --- Embedding model for the watchdog (spec 4.2) ---
    embed_model: str = Field(default="all-MiniLM-L6-v2", validation_alias="VIGIL_EMBED_MODEL")
    # Force the deterministic hashing embedder (offline / tests); skips the ML model entirely.
    embed_hashing: bool = Field(default=False, validation_alias="VIGIL_EMBED_HASHING")

    # --- Optional: goal judge / governor LLM classifier ---
    # provider = "openai" (any OpenAI-compatible endpoint) or "anthropic" (Claude /v1/messages).
    judge_provider: str = Field(default="openai", validation_alias="VIGIL_JUDGE_PROVIDER")
    judge_base_url: str | None = Field(default=None, validation_alias="VIGIL_JUDGE_BASE_URL")
    judge_api_key: str | None = Field(default=None, validation_alias="VIGIL_JUDGE_API_KEY")
    judge_model: str | None = Field(default=None, validation_alias="VIGIL_JUDGE_MODEL")

    # --- Forensics (spec 4.7): content-addressed exchange cache for replay/fork ---
    forensics_enabled: bool = Field(default=True, validation_alias="VIGIL_FORENSICS_ENABLED")

    # --- Effort governor (spec 4.6): opt-in per-step model routing ---
    governor_enabled: bool = Field(default=False, validation_alias="VIGIL_GOVERNOR_ENABLED")
    # Optional JSON override of the provider -> tier -> model routing map.
    governor_model_map: str | None = Field(
        default=None, validation_alias="VIGIL_GOVERNOR_MODEL_MAP"
    )

    # --- Compression Layer 1 (spec 4.5): always-on, free, structural ---
    compress_enabled: bool = Field(default=True, validation_alias="VIGIL_COMPRESS_ENABLED")
    compress_min_tool_bytes: int = Field(
        default=4000, validation_alias="VIGIL_COMPRESS_MIN_TOOL_BYTES"
    )
    compress_floor_messages: int = Field(
        default=6, validation_alias="VIGIL_COMPRESS_FLOOR_MESSAGES"
    )
    compress_dedup_min_run: int = Field(default=3, validation_alias="VIGIL_COMPRESS_DEDUP_MIN_RUN")

    # --- Optional: compression Layer 2 (Token Company) ---
    ttc_api_key: str | None = Field(default=None, validation_alias="TTC_API_KEY")
    ttc_base_url: str | None = Field(default=None, validation_alias="TTC_BASE_URL")

    # --- Optional: semantic cache (Redis LangCache) ---
    redis_langcache_url: str | None = Field(default=None, validation_alias="REDIS_LANGCACHE_URL")
    redis_langcache_api_key: str | None = Field(
        default=None, validation_alias="REDIS_LANGCACHE_API_KEY"
    )
    redis_langcache_cache_id: str | None = Field(
        default=None, validation_alias="REDIS_LANGCACHE_CACHE_ID"
    )
    redis_langcache_min_similarity: float = Field(
        default=0.9, validation_alias="REDIS_LANGCACHE_MIN_SIMILARITY"
    )

    # --- Optional: observability (Phoenix default / Arize AX) ---
    phoenix_collector_endpoint: str | None = Field(
        default=None, validation_alias="PHOENIX_COLLECTOR_ENDPOINT"
    )
    phoenix_api_key: str | None = Field(default=None, validation_alias="PHOENIX_API_KEY")
    arize_space_id: str | None = Field(default=None, validation_alias="ARIZE_SPACE_ID")
    arize_api_key: str | None = Field(default=None, validation_alias="ARIZE_API_KEY")

    # --- Optional: Sentry ---
    sentry_dsn: str | None = Field(default=None, validation_alias="SENTRY_DSN")

    # --- Optional: Orkes webhook ---
    orkes_webhook_url: str | None = Field(default=None, validation_alias="VIGIL_ORKES_WEBHOOK_URL")

    # --- Tracing service name (OpenInference/OTel; exporter chosen by the Phoenix/Arize vars) ---
    tracing_service_name: str = Field(
        default="vigil-proxy", validation_alias="VIGIL_TRACING_SERVICE_NAME"
    )

    @property
    def use_redis(self) -> bool:
        return bool(self.redis_url)

    @property
    def langcache_enabled(self) -> bool:
        return bool(
            self.redis_langcache_url
            and self.redis_langcache_api_key
            and self.redis_langcache_cache_id
        )

    @property
    def tracing_enabled(self) -> bool:
        """Tracing is on when a Phoenix collector or an Arize space is configured."""
        return bool(self.phoenix_collector_endpoint or (self.arize_space_id and self.arize_api_key))

    @property
    def judge_enabled(self) -> bool:
        return bool(self.judge_base_url and self.judge_api_key and self.judge_model)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide singleton so env is read once."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
