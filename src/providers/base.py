"""
Base provider abstract class defining the interface for all LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ProviderStatus(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComplexityLevel(Enum):
    """Request complexity classification."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class Message:
    """Chat message structure."""
    role: str
    content: str


@dataclass
class LLMRequest:
    """Standardized LLM request format."""
    messages: List[Message]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        """Get the full prompt as text for caching/analysis."""
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)

    @property
    def last_user_message(self) -> Optional[str]:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost: float
    cached: bool = False
    cache_type: Optional[str] = None
    trace_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "cached": self.cached,
            "cache_type": self.cache_type,
            "trace_id": self.trace_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ProviderHealth:
    """Provider health status."""
    provider_name: str
    status: ProviderStatus
    latency_ms: Optional[float] = None
    last_check: datetime = field(default_factory=datetime.utcnow)
    error_count: int = 0
    success_count: int = 0
    last_error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.error_count
        if total == 0:
            return 1.0
        return self.success_count / total


@dataclass
class ModelInfo:
    """Model pricing and capability information."""
    model_id: str
    provider: str
    input_cost_per_1k: float  # Cost per 1000 input tokens
    output_cost_per_1k: float  # Cost per 1000 output tokens
    max_tokens: int
    supports_streaming: bool = True
    complexity_level: ComplexityLevel = ComplexityLevel.MODERATE

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str):
        self.name = name
        self._health = ProviderHealth(provider_name=name, status=ProviderStatus.HEALTHY)
        self._models: Dict[str, ModelInfo] = {}

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            request: Standardized LLM request

        Returns:
            LLMResponse with content and metadata

        Raises:
            ProviderError: If the provider fails to generate a response
        """
        pass

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """
        Check provider health.

        Returns:
            ProviderHealth with current status
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of available models.

        Returns:
            List of ModelInfo for all available models
        """
        pass

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for a request.

        Args:
            model: Model ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD
        """
        model_info = self._models.get(model)
        if model_info:
            return model_info.calculate_cost(input_tokens, output_tokens)
        return 0.0

    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Get model information."""
        return self._models.get(model)

    @property
    def health(self) -> ProviderHealth:
        """Get current health status."""
        return self._health

    def record_success(self, latency_ms: float):
        """Record a successful request."""
        self._health.success_count += 1
        self._health.latency_ms = latency_ms
        self._health.last_check = datetime.utcnow()
        if self._health.error_count == 0:
            self._health.status = ProviderStatus.HEALTHY
        elif self._health.success_rate > 0.9:
            self._health.status = ProviderStatus.HEALTHY
        elif self._health.success_rate > 0.5:
            self._health.status = ProviderStatus.DEGRADED

    def record_failure(self, error: str):
        """Record a failed request."""
        self._health.error_count += 1
        self._health.last_error = error
        self._health.last_check = datetime.utcnow()
        if self._health.success_rate < 0.5:
            self._health.status = ProviderStatus.UNHEALTHY
        elif self._health.success_rate < 0.9:
            self._health.status = ProviderStatus.DEGRADED


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, provider: str, message: str, retryable: bool = True):
        self.provider = provider
        self.message = message
        self.retryable = retryable
        super().__init__(f"[{provider}] {message}")


class RateLimitError(ProviderError):
    """Rate limit exceeded error."""

    def __init__(self, provider: str, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__(provider, f"Rate limit exceeded. Retry after: {retry_after}s", retryable=True)


class AuthenticationError(ProviderError):
    """Authentication failed error."""

    def __init__(self, provider: str):
        super().__init__(provider, "Authentication failed", retryable=False)


class InvalidRequestError(ProviderError):
    """Invalid request error."""

    def __init__(self, provider: str, message: str):
        super().__init__(provider, f"Invalid request: {message}", retryable=False)
