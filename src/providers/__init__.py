"""Providers package - LLM provider abstractions."""

from src.providers.base import (
    BaseProvider,
    ProviderStatus,
    ProviderHealth,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    LLMRequest,
    LLMResponse,
    Message,
    ModelInfo,
    ComplexityLevel,
)
from src.providers.groq_provider import GroqProvider
from src.providers.factory import (
    ProviderFactory,
    get_provider,
    get_all_providers,
    get_default_provider,
)
from src.providers.health_monitor import (
    HealthMonitor,
    get_health_monitor,
    start_health_monitoring,
    stop_health_monitoring,
)

__all__ = [
    # Base classes
    "BaseProvider",
    "ProviderStatus",
    "ProviderHealth",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "InvalidRequestError",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "ModelInfo",
    "ComplexityLevel",
    # Providers
    "GroqProvider",
    # Factory
    "ProviderFactory",
    "get_provider",
    "get_all_providers",
    "get_default_provider",
    # Health monitoring
    "HealthMonitor",
    "get_health_monitor",
    "start_health_monitoring",
    "stop_health_monitoring",
]
