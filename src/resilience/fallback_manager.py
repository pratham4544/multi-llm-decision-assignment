"""
Fallback chain management and degraded mode handling.
Orchestrates provider failover and graceful degradation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import structlog

from src.providers.base import (
    BaseProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
    ComplexityLevel,
)
from src.providers.factory import ProviderFactory
from src.cache.cache_manager import get_cache_manager
from src.resilience.circuit_breaker import get_circuit_breaker, CircuitBreakerError
from src.resilience.retry_policy import RetryPolicy, RetryConfig

logger = structlog.get_logger()


@dataclass
class FallbackResult:
    """Result of a fallback chain execution."""
    success: bool
    response: Optional[LLMResponse]
    provider_used: Optional[str]
    model_used: Optional[str]
    attempts: int
    errors: List[Dict[str, Any]]
    degraded_mode: bool
    cached_fallback: bool


class FallbackManager:
    """
    Manages fallback chains and degraded mode operations.

    Responsibilities:
    1. Execute requests through fallback chain
    2. Handle degraded mode (return cached similar response)
    3. Track fallback metrics
    4. Coordinate with circuit breakers
    """

    def __init__(
        self,
        retry_policy: Optional[RetryPolicy] = None,
        enable_degraded_mode: bool = True,
        max_attempts: int = 3
    ):
        """
        Initialize fallback manager.

        Args:
            retry_policy: Retry policy for individual providers
            enable_degraded_mode: Allow cached fallback responses
            max_attempts: Maximum total attempts across providers
        """
        self.retry_policy = retry_policy or RetryPolicy()
        self.enable_degraded_mode = enable_degraded_mode
        self.max_attempts = max_attempts

        # Metrics
        self._fallback_count = 0
        self._degraded_mode_count = 0
        self._total_attempts = 0

    def build_fallback_chain(
        self,
        primary_provider: str,
        primary_model: str,
        complexity: ComplexityLevel
    ) -> List[Tuple[str, str]]:
        """
        Build a fallback chain of (provider, model) pairs.

        Args:
            primary_provider: Primary provider name
            primary_model: Primary model name
            complexity: Request complexity for model selection

        Returns:
            List of (provider, model) tuples in fallback order
        """
        chain = [(primary_provider, primary_model)]

        # Add alternative models from same provider
        provider = ProviderFactory.get_provider(primary_provider)
        if provider:
            for model_info in provider.get_available_models():
                if model_info.model_id != primary_model:
                    chain.append((primary_provider, model_info.model_id))

        # Add other providers
        all_providers = ProviderFactory.get_all_providers()
        for name, prov in all_providers.items():
            if name != primary_provider:
                models = prov.get_available_models()
                if models:
                    # Select model based on complexity
                    for model_info in models:
                        if model_info.complexity_level == complexity:
                            chain.append((name, model_info.model_id))
                            break
                    else:
                        # Just use first model
                        chain.append((name, models[0].model_id))

        return chain[:self.max_attempts]  # Limit chain length

    async def execute_with_fallback(
        self,
        request: LLMRequest,
        fallback_chain: List[Tuple[str, str]],
        trace_id: Optional[str] = None
    ) -> FallbackResult:
        """
        Execute a request with fallback chain.

        Args:
            request: LLM request to execute
            fallback_chain: List of (provider, model) pairs to try
            trace_id: Optional trace ID for logging

        Returns:
            FallbackResult with response or error details
        """
        errors = []
        attempts = 0

        for provider_name, model_name in fallback_chain:
            if attempts >= self.max_attempts:
                break

            attempts += 1
            self._total_attempts += 1

            try:
                response = await self._try_provider(
                    request=request,
                    provider_name=provider_name,
                    model_name=model_name,
                    trace_id=trace_id
                )

                if response:
                    is_fallback = attempts > 1
                    if is_fallback:
                        self._fallback_count += 1

                    return FallbackResult(
                        success=True,
                        response=response,
                        provider_used=provider_name,
                        model_used=model_name,
                        attempts=attempts,
                        errors=errors,
                        degraded_mode=False,
                        cached_fallback=False,
                    )

            except CircuitBreakerError as e:
                errors.append({
                    "provider": provider_name,
                    "model": model_name,
                    "error": str(e),
                    "error_type": "circuit_breaker_open",
                    "timestamp": datetime.utcnow().isoformat(),
                })
                logger.warning(
                    "fallback_circuit_open",
                    trace_id=trace_id,
                    provider=provider_name,
                    reset_time=e.reset_time,
                )
                continue

            except Exception as e:
                errors.append({
                    "provider": provider_name,
                    "model": model_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                logger.warning(
                    "fallback_provider_failed",
                    trace_id=trace_id,
                    provider=provider_name,
                    model=model_name,
                    error=str(e),
                    attempt=attempts,
                )
                continue

        # All providers failed - try degraded mode
        if self.enable_degraded_mode:
            cached_response = await self._try_degraded_mode(request, trace_id)
            if cached_response:
                self._degraded_mode_count += 1
                return FallbackResult(
                    success=True,
                    response=cached_response,
                    provider_used=None,
                    model_used=None,
                    attempts=attempts,
                    errors=errors,
                    degraded_mode=True,
                    cached_fallback=True,
                )

        # Complete failure
        return FallbackResult(
            success=False,
            response=None,
            provider_used=None,
            model_used=None,
            attempts=attempts,
            errors=errors,
            degraded_mode=False,
            cached_fallback=False,
        )

    async def _try_provider(
        self,
        request: LLMRequest,
        provider_name: str,
        model_name: str,
        trace_id: Optional[str] = None
    ) -> Optional[LLMResponse]:
        """
        Try to execute request with a specific provider.

        Args:
            request: LLM request
            provider_name: Provider to use
            model_name: Model to use
            trace_id: Trace ID for logging

        Returns:
            LLMResponse if successful, None otherwise
        """
        provider = ProviderFactory.get_provider(provider_name)
        if not provider:
            raise ProviderError(provider_name, "Provider not available", retryable=False)

        # Check circuit breaker
        circuit_breaker = get_circuit_breaker(provider_name)
        if not circuit_breaker.can_execute():
            raise CircuitBreakerError(
                provider_name,
                circuit_breaker.state,
                circuit_breaker.get_time_until_reset()
            )

        # Update request with model
        request.model = model_name

        # Execute with retry
        try:
            response = await self.retry_policy.execute(
                provider.generate,
                request
            )

            circuit_breaker.record_success()

            logger.info(
                "fallback_provider_success",
                trace_id=trace_id,
                provider=provider_name,
                model=model_name,
            )

            return response

        except Exception as e:
            circuit_breaker.record_failure(str(e))
            raise

    async def _try_degraded_mode(
        self,
        request: LLMRequest,
        trace_id: Optional[str] = None
    ) -> Optional[LLMResponse]:
        """
        Try to return a cached similar response in degraded mode.

        Args:
            request: LLM request
            trace_id: Trace ID for logging

        Returns:
            Cached LLMResponse if available, None otherwise
        """
        cache_manager = get_cache_manager()

        # Try semantic cache with lower threshold
        if cache_manager.semantic_cache:
            # Temporarily lower threshold for degraded mode
            original_threshold = cache_manager.semantic_cache.similarity_threshold
            cache_manager.semantic_cache.similarity_threshold = 0.8

            try:
                result = cache_manager.semantic_cache.find_similar(request)
                if result:
                    response, similarity = result
                    response.metadata["degraded_mode"] = True
                    response.metadata["similarity_score"] = similarity

                    logger.info(
                        "degraded_mode_cache_hit",
                        trace_id=trace_id,
                        similarity=round(similarity, 4),
                    )

                    return response

            finally:
                # Restore original threshold
                cache_manager.semantic_cache.similarity_threshold = original_threshold

        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get fallback manager metrics."""
        return {
            "total_attempts": self._total_attempts,
            "fallback_count": self._fallback_count,
            "degraded_mode_count": self._degraded_mode_count,
            "fallback_rate": (
                self._fallback_count / self._total_attempts
                if self._total_attempts > 0 else 0.0
            ),
        }


# Global fallback manager
_fallback_manager: Optional[FallbackManager] = None


def get_fallback_manager() -> FallbackManager:
    """Get the global fallback manager."""
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = FallbackManager()
    return _fallback_manager
