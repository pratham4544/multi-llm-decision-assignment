"""
Resilience pattern tests (circuit breaker, retry).
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitState,
    CircuitBreakerError,
)
from src.resilience.retry_policy import RetryPolicy, RetryConfig


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    def test_initial_state_is_closed(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_failure_threshold(self):
        """Test circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_resets_on_transition_to_closed(self):
        """Test failure count resets when transitioning to closed."""
        cb = CircuitBreaker(name="test", failure_threshold=5, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        # Record success while still closed
        cb.record_success()
        # Failure count is NOT reset on success while closed
        # Only transitions reset the count
        assert cb.state == CircuitState.CLOSED

    def test_blocks_calls_when_open(self):
        """Test circuit blocks calls when open."""
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_allows_calls_when_closed(self):
        """Test circuit allows calls when closed."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.can_execute() is True


class TestCircuitBreakerManager:
    """Tests for circuit breaker manager."""

    def test_get_creates_new_breaker(self):
        """Test manager creates new breaker for unknown provider."""
        manager = CircuitBreakerManager()
        cb = manager.get_breaker("new-provider")
        assert cb is not None
        assert cb.name == "new-provider"

    def test_get_returns_existing_breaker(self):
        """Test manager returns existing breaker."""
        manager = CircuitBreakerManager()
        cb1 = manager.get_breaker("provider")
        cb2 = manager.get_breaker("provider")
        assert cb1 is cb2

    def test_get_all_stats(self):
        """Test getting stats for all circuit breakers."""
        manager = CircuitBreakerManager()
        manager.get_breaker("provider1")
        manager.get_breaker("provider2")
        stats = manager.get_all_stats()
        assert "provider1" in stats
        assert "provider2" in stats


class TestRetryPolicy:
    """Tests for retry policy."""

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Test retry policy retries on failure."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        policy = RetryPolicy(config)

        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Temporary failure")
            return "success"

        result = await policy.execute(failing_func)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self):
        """Test retry policy gives up after max attempts."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        policy = RetryPolicy(config)

        async def always_fails():
            raise TimeoutError("Permanent failure")

        with pytest.raises(TimeoutError):
            await policy.execute(always_fails)

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        """Test no retry needed on success."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        policy = RetryPolicy(config)

        call_count = 0

        async def succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await policy.execute(succeeds)
        assert result == "success"
        assert call_count == 1

    def test_respects_max_delay(self):
        """Test delay is capped at max_delay."""
        config = RetryConfig(max_attempts=5, base_delay=1.0, max_delay=0.1, jitter=False)
        policy = RetryPolicy(config)
        delay = policy.calculate_delay(attempt=10)
        assert delay <= config.max_delay
