"""
FastAPI dependency injection functions.
Provides reusable dependencies for routes.
"""

from typing import Optional
import uuid
from fastapi import Depends, Header, HTTPException, Request, status
import structlog

from src.config import settings
from src.rate_limit.token_bucket import TokenBucket, RateLimitResult, get_rate_limiter
from src.cache.cache_manager import CacheManager, get_cache_manager
from src.providers.health_monitor import HealthMonitor, get_health_monitor

logger = structlog.get_logger()


def get_client_id(
    x_client_id: Optional[str] = Header(None, alias="X-Client-ID"),
    request: Request = None
) -> str:
    """
    Extract client ID from request.

    Priority:
    1. X-Client-ID header
    2. Client IP address

    Returns:
        Client identifier string
    """
    if x_client_id:
        return x_client_id

    # Fall back to client IP
    if request and request.client:
        return request.client.host

    return "anonymous"


def get_client_tier(
    x_client_tier: Optional[str] = Header(None, alias="X-Client-Tier"),
) -> str:
    """
    Get client tier from request.

    Tiers:
    - standard: Default tier
    - premium: Higher rate limits
    - enterprise: Highest rate limits

    Returns:
        Tier string
    """
    if x_client_tier and x_client_tier in ("standard", "premium", "enterprise"):
        return x_client_tier
    return "standard"


def get_trace_id(
    x_trace_id: Optional[str] = Header(None, alias="X-Trace-ID"),
) -> str:
    """
    Get or generate trace ID.

    Returns:
        Trace ID string
    """
    if x_trace_id:
        return x_trace_id
    return str(uuid.uuid4())


async def check_rate_limit(
    client_id: str = Depends(get_client_id),
    tier: str = Depends(get_client_tier),
) -> RateLimitResult:
    """
    Check rate limit for client.

    Raises:
        HTTPException: If rate limit exceeded

    Returns:
        RateLimitResult with remaining quota
    """
    rate_limiter = get_rate_limiter()
    result = rate_limiter.check(client_id, tier)

    if not result.allowed:
        logger.warning(
            "rate_limit_rejected",
            client_id=client_id,
            tier=tier,
            retry_after=result.retry_after,
        )

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "message": f"Rate limit exceeded. Retry after {result.retry_after:.1f} seconds.",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
            headers={
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": str(result.remaining),
                "X-RateLimit-Reset": str(int(result.reset_time)),
                "Retry-After": str(int(result.retry_after or 60)),
            }
        )

    return result


def rate_limit_headers(result: RateLimitResult) -> dict:
    """Generate rate limit headers for response."""
    return {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
        "X-RateLimit-Reset": str(int(result.reset_time)),
    }


class RequestContext:
    """Request context with client info and trace ID."""

    def __init__(
        self,
        client_id: str,
        tier: str,
        trace_id: str,
        rate_limit: RateLimitResult
    ):
        self.client_id = client_id
        self.tier = tier
        self.trace_id = trace_id
        self.rate_limit = rate_limit


async def get_request_context(
    client_id: str = Depends(get_client_id),
    tier: str = Depends(get_client_tier),
    trace_id: str = Depends(get_trace_id),
    rate_limit: RateLimitResult = Depends(check_rate_limit),
) -> RequestContext:
    """
    Build complete request context.

    Returns:
        RequestContext with all request metadata
    """
    return RequestContext(
        client_id=client_id,
        tier=tier,
        trace_id=trace_id,
        rate_limit=rate_limit
    )


# Service dependencies

def get_cache_service() -> CacheManager:
    """Get cache manager service."""
    return get_cache_manager()


def get_health_service() -> HealthMonitor:
    """Get health monitor service."""
    return get_health_monitor()


def get_rate_limit_service() -> TokenBucket:
    """Get rate limiter service."""
    return get_rate_limiter()
