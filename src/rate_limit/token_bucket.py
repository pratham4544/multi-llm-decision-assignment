"""
Token bucket algorithm for rate limiting.
Per-client rate limiting using Redis for distributed state.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import redis
import structlog

from src.config import settings

logger = structlog.get_logger()


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    limit: int
    reset_time: float
    retry_after: Optional[float] = None


class TokenBucket:
    """
    Token bucket rate limiter with Redis backend.

    Each client has a bucket that:
    - Holds up to `capacity` tokens
    - Refills at `refill_rate` tokens per second
    - Each request consumes 1 token

    Redis keys:
    - rate_limit:{client_id}:tokens - Current token count
    - rate_limit:{client_id}:last_update - Last update timestamp
    """

    KEY_PREFIX = "rate_limit:"

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        default_capacity: int = 1000,
        default_refill_rate: float = 0.28,  # ~1000/hour
    ):
        """
        Initialize token bucket rate limiter.

        Args:
            redis_client: Redis client for distributed state
            default_capacity: Maximum tokens per bucket
            default_refill_rate: Tokens added per second
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

        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate

        # Tier configurations (capacity, refill_rate)
        self.tier_configs = {
            "standard": (settings.rate_limit_tier1, settings.rate_limit_tier1 / 3600),
            "premium": (settings.rate_limit_tier2, settings.rate_limit_tier2 / 3600),
            "enterprise": (settings.rate_limit_tier3, settings.rate_limit_tier3 / 3600),
        }

    def _get_keys(self, client_id: str) -> Tuple[str, str]:
        """Get Redis keys for a client."""
        tokens_key = f"{self.KEY_PREFIX}{client_id}:tokens"
        time_key = f"{self.KEY_PREFIX}{client_id}:last_update"
        return tokens_key, time_key

    def _get_tier_config(self, tier: str) -> Tuple[int, float]:
        """Get capacity and refill rate for a tier."""
        return self.tier_configs.get(tier, (self.default_capacity, self.default_refill_rate))

    def check(
        self,
        client_id: str,
        tier: str = "standard",
        tokens_needed: int = 1
    ) -> RateLimitResult:
        """
        Check if a request is allowed and consume tokens.

        This is an atomic operation that:
        1. Calculates refilled tokens since last update
        2. Checks if enough tokens are available
        3. Consumes tokens if allowed
        4. Returns the result

        Args:
            client_id: Client identifier
            tier: Client tier (standard/premium/enterprise)
            tokens_needed: Tokens to consume (default 1)

        Returns:
            RateLimitResult with allowed status and remaining tokens
        """
        tokens_key, time_key = self._get_keys(client_id)
        capacity, refill_rate = self._get_tier_config(tier)

        now = time.time()

        # Use Lua script for atomic operation
        script = """
        local tokens_key = KEYS[1]
        local time_key = KEYS[2]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local tokens_needed = tonumber(ARGV[4])

        -- Get current state
        local tokens = tonumber(redis.call('GET', tokens_key)) or capacity
        local last_update = tonumber(redis.call('GET', time_key)) or now

        -- Calculate refilled tokens
        local elapsed = now - last_update
        local refilled = elapsed * refill_rate
        tokens = math.min(capacity, tokens + refilled)

        -- Check if request is allowed
        if tokens >= tokens_needed then
            tokens = tokens - tokens_needed
            redis.call('SET', tokens_key, tokens)
            redis.call('SET', time_key, now)
            redis.call('EXPIRE', tokens_key, 7200)
            redis.call('EXPIRE', time_key, 7200)
            return {1, math.floor(tokens), capacity}
        else
            -- Calculate retry after
            local tokens_deficit = tokens_needed - tokens
            local retry_after = tokens_deficit / refill_rate
            return {0, math.floor(tokens), capacity, retry_after}
        end
        """

        try:
            result = self.redis.eval(
                script,
                2,
                tokens_key,
                time_key,
                capacity,
                refill_rate,
                now,
                tokens_needed
            )

            allowed = result[0] == 1
            remaining = int(result[1])
            limit = int(result[2])
            retry_after = float(result[3]) if len(result) > 3 else None

            # Calculate reset time (when bucket will be full)
            if remaining < capacity:
                tokens_to_refill = capacity - remaining
                reset_time = now + (tokens_to_refill / refill_rate)
            else:
                reset_time = now

            if not allowed:
                logger.warning(
                    "rate_limit_exceeded",
                    client_id=client_id,
                    tier=tier,
                    remaining=remaining,
                    retry_after=round(retry_after, 2) if retry_after else None,
                )

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                limit=limit,
                reset_time=reset_time,
                retry_after=retry_after,
            )

        except Exception as e:
            logger.error("rate_limit_check_error", error=str(e), client_id=client_id)
            # Fail open - allow request on error
            return RateLimitResult(
                allowed=True,
                remaining=capacity,
                limit=capacity,
                reset_time=now,
            )

    def get_status(self, client_id: str, tier: str = "standard") -> RateLimitResult:
        """
        Get current rate limit status without consuming tokens.

        Args:
            client_id: Client identifier
            tier: Client tier

        Returns:
            RateLimitResult with current status
        """
        tokens_key, time_key = self._get_keys(client_id)
        capacity, refill_rate = self._get_tier_config(tier)

        now = time.time()

        try:
            tokens = self.redis.get(tokens_key)
            last_update = self.redis.get(time_key)

            if tokens is None:
                return RateLimitResult(
                    allowed=True,
                    remaining=capacity,
                    limit=capacity,
                    reset_time=now,
                )

            tokens = float(tokens)
            last_update = float(last_update) if last_update else now

            # Calculate refilled tokens
            elapsed = now - last_update
            refilled = elapsed * refill_rate
            current_tokens = min(capacity, tokens + refilled)

            if current_tokens < capacity:
                tokens_to_refill = capacity - current_tokens
                reset_time = now + (tokens_to_refill / refill_rate)
            else:
                reset_time = now

            return RateLimitResult(
                allowed=current_tokens >= 1,
                remaining=int(current_tokens),
                limit=capacity,
                reset_time=reset_time,
            )

        except Exception as e:
            logger.error("rate_limit_status_error", error=str(e))
            return RateLimitResult(
                allowed=True,
                remaining=capacity,
                limit=capacity,
                reset_time=now,
            )

    def reset(self, client_id: str):
        """Reset rate limit for a client (for testing/admin)."""
        tokens_key, time_key = self._get_keys(client_id)
        self.redis.delete(tokens_key, time_key)

    def get_all_limits(self) -> Dict[str, RateLimitResult]:
        """Get rate limit status for all tracked clients."""
        results = {}
        try:
            keys = self.redis.keys(f"{self.KEY_PREFIX}*:tokens")
            for key in keys:
                client_id = key.replace(f"{self.KEY_PREFIX}", "").replace(":tokens", "")
                results[client_id] = self.get_status(client_id)
        except Exception:
            pass
        return results


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        rate_limiter: Optional[TokenBucket] = None,
        get_client_id: callable = None,
        get_tier: callable = None
    ):
        """
        Initialize rate limit middleware.

        Args:
            rate_limiter: TokenBucket instance
            get_client_id: Function to extract client ID from request
            get_tier: Function to get client tier
        """
        self.rate_limiter = rate_limiter or TokenBucket()
        self.get_client_id = get_client_id or (lambda r: r.client.host)
        self.get_tier = get_tier or (lambda r: "standard")


# Global rate limiter
_rate_limiter: Optional[TokenBucket] = None


def get_rate_limiter() -> TokenBucket:
    """Get the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = TokenBucket()
    return _rate_limiter
