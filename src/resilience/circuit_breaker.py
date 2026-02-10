"""
Circuit breaker implementation for provider fault tolerance.
Prevents cascading failures by temporarily blocking failing providers.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar
import asyncio
import functools
import structlog

from src.config import settings

logger = structlog.get_logger()

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing recovery, limited requests


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    last_state_change: datetime
    total_requests: int
    blocked_requests: int

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.failure_count / total


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, name: str, state: CircuitState, reset_time: Optional[float] = None):
        self.name = name
        self.state = state
        self.reset_time = reset_time
        message = f"Circuit breaker '{name}' is {state.value}"
        if reset_time:
            message += f", resets in {reset_time:.1f}s"
        super().__init__(message)


class CircuitBreaker:
    """
    Circuit breaker for a single provider.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: After failure_threshold failures, blocks all requests
    - HALF_OPEN: After timeout, allows one test request

    Transitions:
    - CLOSED -> OPEN: failure_count >= failure_threshold
    - OPEN -> HALF_OPEN: timeout elapsed
    - HALF_OPEN -> CLOSED: test request succeeds
    - HALF_OPEN -> OPEN: test request fails
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = None,
        recovery_timeout: int = None,
        half_open_max_calls: int = 1
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name identifier (usually provider name)
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold or settings.circuit_breaker_failure_threshold
        self.recovery_timeout = recovery_timeout or settings.circuit_breaker_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._last_state_change = time.time()
        self._total_requests = 0
        self._blocked_requests = 0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for automatic transitions."""
        if self._state == CircuitState.OPEN:
            # Check if timeout has elapsed
            elapsed = time.time() - self._last_state_change
            if elapsed >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

        logger.info(
            "circuit_state_change",
            circuit=self.name,
            from_state=old_state.value,
            to_state=new_state.value,
        )

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        state = self.state  # This may trigger OPEN -> HALF_OPEN transition

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        if state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls

        return False

    def record_success(self):
        """Record a successful call."""
        self._success_count += 1
        self._last_success_time = time.time()
        self._total_requests += 1

        if self._state == CircuitState.HALF_OPEN:
            # Successful test call, close the circuit
            self._transition_to(CircuitState.CLOSED)

    def record_failure(self, error: Optional[str] = None):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        self._total_requests += 1

        if self._state == CircuitState.HALF_OPEN:
            # Test call failed, reopen the circuit
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "circuit_opened",
                    circuit=self.name,
                    failure_count=self._failure_count,
                    last_error=error,
                )

    def get_time_until_reset(self) -> Optional[float]:
        """Get seconds until circuit might close."""
        if self._state != CircuitState.OPEN:
            return None
        elapsed = time.time() - self._last_state_change
        remaining = self.recovery_timeout - elapsed
        return max(0, remaining)

    def get_stats(self) -> CircuitStats:
        """Get current statistics."""
        return CircuitStats(
            state=self.state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=datetime.fromtimestamp(self._last_failure_time)
            if self._last_failure_time else None,
            last_success_time=datetime.fromtimestamp(self._last_success_time)
            if self._last_success_time else None,
            last_state_change=datetime.fromtimestamp(self._last_state_change),
            total_requests=self._total_requests,
            blocked_requests=self._blocked_requests,
        )

    def reset(self):
        """Reset the circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_success_time = None
        self._last_state_change = time.time()
        self._half_open_calls = 0

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the function

        Raises:
            CircuitBreakerError: If circuit is open
        """
        async with self._lock:
            if not self.can_execute():
                self._blocked_requests += 1
                raise CircuitBreakerError(
                    self.name,
                    self._state,
                    self.get_time_until_reset()
                )

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(str(e))
            raise


class CircuitBreakerManager:
    """Manages circuit breakers for multiple providers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name)
        return self._breakers[name]

    def get_all_stats(self) -> Dict[str, CircuitStats]:
        """Get stats for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global circuit breaker manager
_cb_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager."""
    global _cb_manager
    if _cb_manager is None:
        _cb_manager = CircuitBreakerManager()
    return _cb_manager


def get_circuit_breaker(provider: str) -> CircuitBreaker:
    """Get circuit breaker for a provider."""
    return get_circuit_breaker_manager().get_breaker(provider)
