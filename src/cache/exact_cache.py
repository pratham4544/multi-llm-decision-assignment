"""
Exact match caching implementation using hash-based lookup.
Uses SHA256 hash of request content for cache keys.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Optional
import redis
import structlog

from src.config import settings
from src.providers.base import LLMRequest, LLMResponse, Message

logger = structlog.get_logger()


class ExactCache:
    """Hash-based exact match cache using Redis."""

    CACHE_PREFIX = "llm:cache:exact:"
    STATS_KEY = "llm:cache:exact:stats"

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        ttl: Optional[int] = None
    ):
        """
        Initialize exact cache.

        Args:
            redis_client: Redis client instance (creates one if not provided)
            ttl: Cache TTL in seconds (defaults to settings.cache_ttl)
        """
        self.ttl = ttl or settings.cache_ttl

        if redis_client:
            self.redis = redis_client
        else:
            self.redis = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password or None,
                decode_responses=True
            )

        self._hits = 0
        self._misses = 0

    def _generate_cache_key(self, request: LLMRequest) -> str:
        """
        Generate a cache key from request content.

        The key is based on:
        - Messages content and roles
        - Model (if specified)
        - Temperature

        Args:
            request: LLM request

        Returns:
            Cache key string
        """
        # Build a canonical representation
        key_data = {
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ],
            "model": request.model,
            "temperature": request.temperature,
        }

        # Create deterministic JSON string
        key_json = json.dumps(key_data, sort_keys=True, separators=(",", ":"))

        # Hash for fixed-length key
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()

        return f"{self.CACHE_PREFIX}{key_hash}"

    def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """
        Get cached response for a request.

        Args:
            request: LLM request to look up

        Returns:
            Cached LLMResponse or None if not found
        """
        cache_key = self._generate_cache_key(request)

        try:
            cached_data = self.redis.get(cache_key)

            if cached_data:
                self._hits += 1
                self._update_stats("hits")

                data = json.loads(cached_data)

                logger.debug(
                    "cache_hit",
                    cache_type="exact",
                    key=cache_key[-12:],  # Last 12 chars for logging
                )

                return LLMResponse(
                    content=data["content"],
                    model=data["model"],
                    provider=data["provider"],
                    input_tokens=data["input_tokens"],
                    output_tokens=data["output_tokens"],
                    total_tokens=data["total_tokens"],
                    latency_ms=0.0,  # Cached response has no latency
                    cost=0.0,  # Cached response has no cost
                    cached=True,
                    cache_type="exact",
                    created_at=datetime.fromisoformat(data["created_at"]),
                    metadata=data.get("metadata", {}),
                )

            self._misses += 1
            self._update_stats("misses")
            return None

        except Exception as e:
            logger.error("cache_get_error", error=str(e), cache_type="exact")
            self._misses += 1
            return None

    def set(
        self,
        request: LLMRequest,
        response: LLMResponse,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a response for a request.

        Args:
            request: Original LLM request
            response: LLM response to cache
            ttl: Optional TTL override

        Returns:
            True if cached successfully
        """
        cache_key = self._generate_cache_key(request)
        cache_ttl = ttl or self.ttl

        try:
            cache_data = {
                "content": response.content,
                "model": response.model,
                "provider": response.provider,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "created_at": response.created_at.isoformat(),
                "metadata": response.metadata,
            }

            self.redis.setex(
                cache_key,
                cache_ttl,
                json.dumps(cache_data)
            )

            logger.debug(
                "cache_set",
                cache_type="exact",
                key=cache_key[-12:],
                ttl=cache_ttl
            )

            return True

        except Exception as e:
            logger.error("cache_set_error", error=str(e), cache_type="exact")
            return False

    def exists(self, request: LLMRequest) -> bool:
        """Check if a cache entry exists for the request."""
        cache_key = self._generate_cache_key(request)
        try:
            return bool(self.redis.exists(cache_key))
        except Exception:
            return False

    def delete(self, request: LLMRequest) -> bool:
        """Delete a cache entry."""
        cache_key = self._generate_cache_key(request)
        try:
            return bool(self.redis.delete(cache_key))
        except Exception:
            return False

    def clear(self) -> int:
        """Clear all exact cache entries."""
        try:
            keys = self.redis.keys(f"{self.CACHE_PREFIX}*")
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error("cache_clear_error", error=str(e))
            return 0

    def _update_stats(self, stat_type: str):
        """Update cache statistics in Redis."""
        try:
            self.redis.hincrby(self.STATS_KEY, stat_type, 1)
            self.redis.hincrby(self.STATS_KEY, "total", 1)
        except Exception:
            pass  # Stats are best-effort

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats = self.redis.hgetall(self.STATS_KEY)
            hits = int(stats.get("hits", 0))
            misses = int(stats.get("misses", 0))
            total = hits + misses

            return {
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": hits / total if total > 0 else 0.0,
                "local_hits": self._hits,
                "local_misses": self._misses,
            }
        except Exception:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total": self._hits + self._misses,
                "hit_rate": self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0 else 0.0,
            }

    @property
    def hit_rate(self) -> float:
        """Calculate local hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
