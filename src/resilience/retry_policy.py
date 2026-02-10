"""
Retry logic with exponential backoff.
Handles transient failures with configurable retry policies.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Type, TypeVar
import structlog

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

from src.config import settings
from src.providers.base import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
)

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (RateLimitError, TimeoutError, ConnectionError)
    non_retryable_exceptions: tuple = (AuthenticationError, InvalidRequestError)


class RetryPolicy:
    """
    Configurable retry policy with exponential backoff.

    Features:
    - Exponential backoff with optional jitter
    - Configurable max attempts
    - Exception type filtering
    - Logging of retry attempts
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry policy.

        Args:
            config: Retry configuration (uses defaults if not provided)
        """
        self.config = config or RetryConfig(
            max_attempts=settings.max_retries,
            base_delay=settings.retry_backoff_base,
        )

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.

        Args:
            attempt: Attempt number (1-based)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base * (exponential_base ^ (attempt - 1))
        delay = self.config.base_delay * (
            self.config.exponential_base ** (attempt - 1)
        )

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter (0-50% of delay)
        if self.config.jitter:
            jitter = delay * random.uniform(0, 0.5)
            delay += jitter

        return delay

    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exception: The exception that occurred

        Returns:
            True if should retry
        """
        # Never retry non-retryable exceptions
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False

        # Retry known retryable exceptions
        if isinstance(exception, self.config.retryable_exceptions):
            return True

        # Retry generic provider errors if marked retryable
        if isinstance(exception, ProviderError):
            return exception.retryable

        # Don't retry unknown exceptions by default
        return False

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute a function with retry logic.

        Args:
            func: Async function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the function

        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        attempt = 0

        while attempt < self.config.max_attempts:
            attempt += 1

            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if not self.should_retry(e):
                    logger.debug(
                        "retry_not_retryable",
                        attempt=attempt,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

                if attempt >= self.config.max_attempts:
                    logger.warning(
                        "retry_exhausted",
                        max_attempts=self.config.max_attempts,
                        error=str(e),
                    )
                    raise

                delay = self.calculate_delay(attempt)

                logger.info(
                    "retry_attempt",
                    attempt=attempt,
                    max_attempts=self.config.max_attempts,
                    delay=round(delay, 2),
                    error=str(e),
                    error_type=type(e).__name__,
                )

                await asyncio.sleep(delay)

        raise last_exception


def with_retry(
    max_attempts: int = None,
    base_delay: float = None,
    retryable_exceptions: tuple = None
):
    """
    Decorator for adding retry logic to async functions.

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        retryable_exceptions: Tuple of exceptions to retry

    Returns:
        Decorated function
    """
    config = RetryConfig(
        max_attempts=max_attempts or settings.max_retries,
        base_delay=base_delay or settings.retry_backoff_base,
        retryable_exceptions=retryable_exceptions or (
            RateLimitError, TimeoutError, ConnectionError
        ),
    )
    policy = RetryPolicy(config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args, **kwargs) -> T:
            return await policy.execute(func, *args, **kwargs)
        return wrapper

    return decorator


def create_tenacity_retry(
    max_attempts: int = None,
    base_delay: float = None,
    max_delay: float = 10.0,
):
    """
    Create a tenacity retry decorator for more advanced use cases.

    Args:
        max_attempts: Maximum attempts
        base_delay: Base delay for exponential backoff
        max_delay: Maximum delay

    Returns:
        tenacity retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts or settings.max_retries),
        wait=wait_exponential(
            multiplier=base_delay or settings.retry_backoff_base,
            max=max_delay
        ),
        retry=retry_if_exception_type((RateLimitError, TimeoutError, ConnectionError)),
        before_sleep=before_sleep_log(logger, log_level=20),  # INFO level
        reraise=True,
    )


# Default retry policy
_default_policy: Optional[RetryPolicy] = None


def get_default_retry_policy() -> RetryPolicy:
    """Get the default retry policy."""
    global _default_policy
    if _default_policy is None:
        _default_policy = RetryPolicy()
    return _default_policy
