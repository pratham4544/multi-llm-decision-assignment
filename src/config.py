"""
Application configuration using Pydantic Settings.
Loads configuration from environment variables (.env file).
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Environment
    env: str = Field(default="development", description="Environment (development/staging/production)")
    debug: bool = Field(default=True, description="Debug mode")

    # API Keys
    groq_api_key: str = Field(..., description="Groq API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")

    # Redis
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")

    # FastAPI
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # Rate Limiting (requests per hour)
    rate_limit_tier1: int = Field(default=1000, description="Tier 1 rate limit")
    rate_limit_tier2: int = Field(default=10000, description="Tier 2 rate limit")
    rate_limit_tier3: int = Field(default=100000, description="Tier 3 rate limit")

    # Cache
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    semantic_similarity_threshold: float = Field(default=0.95, description="Semantic similarity threshold")

    # Workers
    min_workers: int = Field(default=3, description="Minimum worker count")
    max_workers: int = Field(default=20, description="Maximum worker count")
    worker_timeout: int = Field(default=300, description="Worker timeout in seconds")

    # Circuit Breaker
    circuit_breaker_failure_threshold: int = Field(default=3, description="Circuit breaker failure threshold")
    circuit_breaker_timeout: int = Field(default=30, description="Circuit breaker timeout in seconds")

    # Retry
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_backoff_base: float = Field(default=0.1, description="Retry backoff base in seconds")

    # LangSmith (optional)
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith tracing")
    langchain_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langchain_project: str = Field(default="llm-router-system", description="LangSmith project name")

    # Embeddings
    use_fake_embeddings: bool = Field(default=True, description="Use fake embeddings for testing")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")

    @property
    def redis_url(self) -> str:
        """Get Redis URL for connection."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.env.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()
