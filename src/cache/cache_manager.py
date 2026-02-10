"""
Unified cache manager interface.
Orchestrates exact and semantic caching.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
import redis
import structlog

from src.config import settings
from src.cache.exact_cache import ExactCache
from src.cache.semantic_cache import SemanticCache
from src.providers.base import LLMRequest, LLMResponse

logger = structlog.get_logger()


@dataclass
class CacheResult:
    """Cache lookup result."""
    hit: bool
    response: Optional[LLMResponse]
    cache_type: Optional[str]  # "exact" or "semantic"
    similarity_score: Optional[float]  # For semantic cache
    lookup_time_ms: float


class CacheManager:
    """
    Unified cache manager that orchestrates multi-level caching.

    Cache lookup order:
    1. Exact match (fast, hash-based)
    2. Semantic similarity (slower, embedding-based)
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        enable_exact: bool = True,
        enable_semantic: bool = True,
        ttl: Optional[int] = None,
        semantic_threshold: Optional[float] = None
    ):
        """
        Initialize cache manager.

        Args:
            redis_client: Shared Redis client
            enable_exact: Enable exact match caching
            enable_semantic: Enable semantic caching
            ttl: Cache TTL in seconds
            semantic_threshold: Semantic similarity threshold
        """
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

        self.enable_exact = enable_exact
        self.enable_semantic = enable_semantic
        self.ttl = ttl or settings.cache_ttl

        # Initialize cache layers
        self.exact_cache = ExactCache(
            redis_client=self.redis,
            ttl=self.ttl
        ) if enable_exact else None

        self.semantic_cache = SemanticCache(
            redis_client=self.redis,
            ttl=self.ttl,
            similarity_threshold=semantic_threshold
        ) if enable_semantic else None

    def get(self, request: LLMRequest) -> CacheResult:
        """
        Look up a cached response using multi-level caching.

        Args:
            request: LLM request to look up

        Returns:
            CacheResult with hit status and response
        """
        import time
        start_time = time.time()

        # Level 1: Exact match
        if self.exact_cache:
            response = self.exact_cache.get(request)
            if response:
                lookup_time = (time.time() - start_time) * 1000
                logger.info(
                    "cache_hit",
                    cache_type="exact",
                    lookup_time_ms=round(lookup_time, 2)
                )
                return CacheResult(
                    hit=True,
                    response=response,
                    cache_type="exact",
                    similarity_score=1.0,
                    lookup_time_ms=lookup_time
                )

        # Level 2: Semantic similarity
        if self.semantic_cache:
            result = self.semantic_cache.find_similar(request)
            if result:
                response, similarity = result
                lookup_time = (time.time() - start_time) * 1000
                logger.info(
                    "cache_hit",
                    cache_type="semantic",
                    similarity=round(similarity, 4),
                    lookup_time_ms=round(lookup_time, 2)
                )
                return CacheResult(
                    hit=True,
                    response=response,
                    cache_type="semantic",
                    similarity_score=similarity,
                    lookup_time_ms=lookup_time
                )

        # Cache miss
        lookup_time = (time.time() - start_time) * 1000
        logger.debug("cache_miss", lookup_time_ms=round(lookup_time, 2))

        return CacheResult(
            hit=False,
            response=None,
            cache_type=None,
            similarity_score=None,
            lookup_time_ms=lookup_time
        )

    def set(
        self,
        request: LLMRequest,
        response: LLMResponse,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a response in both exact and semantic caches.

        Args:
            request: Original request
            response: Response to cache
            ttl: Optional TTL override

        Returns:
            True if cached successfully in at least one layer
        """
        success = False

        # Add to exact cache
        if self.exact_cache:
            if self.exact_cache.set(request, response, ttl):
                success = True

        # Add to semantic cache
        if self.semantic_cache:
            if self.semantic_cache.set(request, response, ttl):
                success = True

        return success

    def invalidate(self, request: LLMRequest) -> bool:
        """
        Invalidate cache entries for a request.

        Note: Only exact cache can be invalidated directly.
        Semantic cache entries expire via TTL.

        Args:
            request: Request to invalidate

        Returns:
            True if invalidation was successful
        """
        if self.exact_cache:
            return self.exact_cache.delete(request)
        return False

    def clear(self) -> Dict[str, int]:
        """
        Clear all caches.

        Returns:
            Dict with count of cleared entries per cache type
        """
        cleared = {}

        if self.exact_cache:
            cleared["exact"] = self.exact_cache.clear()

        if self.semantic_cache:
            cleared["semantic"] = self.semantic_cache.clear()

        logger.info("cache_cleared", cleared=cleared)
        return cleared

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all cache layers."""
        stats = {
            "enabled": {
                "exact": self.enable_exact,
                "semantic": self.enable_semantic,
            },
            "ttl": self.ttl,
        }

        if self.exact_cache:
            stats["exact"] = self.exact_cache.get_stats()

        if self.semantic_cache:
            stats["semantic"] = self.semantic_cache.get_stats()

        # Calculate combined stats
        total_hits = 0
        total_misses = 0

        if self.exact_cache:
            exact_stats = stats.get("exact", {})
            total_hits += exact_stats.get("hits", 0)

        if self.semantic_cache:
            semantic_stats = stats.get("semantic", {})
            total_hits += semantic_stats.get("hits", 0)
            total_misses = semantic_stats.get("misses", 0)  # Use semantic misses as total

        stats["combined"] = {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_requests": total_hits + total_misses,
            "overall_hit_rate": (
                total_hits / (total_hits + total_misses)
                if (total_hits + total_misses) > 0 else 0.0
            ),
        }

        return stats

    def get_savings_estimate(self) -> Dict[str, Any]:
        """
        Estimate cost savings from caching.

        Returns:
            Dict with savings estimates
        """
        stats = self.get_stats()
        combined = stats.get("combined", {})

        # Assume average cost per request (in USD)
        avg_cost_per_request = 0.001  # $0.001 average

        total_hits = combined.get("total_hits", 0)
        estimated_savings = total_hits * avg_cost_per_request

        return {
            "cache_hits": total_hits,
            "estimated_cost_saved_usd": round(estimated_savings, 4),
            "hit_rate": combined.get("overall_hit_rate", 0.0),
        }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def reset_cache_manager():
    """Reset the global cache manager (for testing)."""
    global _cache_manager
    _cache_manager = None
