"""Resilience package - Fault tolerance and recovery."""

from src.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitBreakerError,
    CircuitState,
    CircuitStats,
    get_circuit_breaker_manager,
    get_circuit_breaker,
)
from src.resilience.retry_policy import (
    RetryPolicy,
    RetryConfig,
    with_retry,
    get_default_retry_policy,
)
from src.resilience.fallback_manager import (
    FallbackManager,
    FallbackResult,
    get_fallback_manager,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerManager",
    "CircuitBreakerError",
    "CircuitState",
    "CircuitStats",
    "get_circuit_breaker_manager",
    "get_circuit_breaker",
    # Retry
    "RetryPolicy",
    "RetryConfig",
    "with_retry",
    "get_default_retry_policy",
    # Fallback
    "FallbackManager",
    "FallbackResult",
    "get_fallback_manager",
]
