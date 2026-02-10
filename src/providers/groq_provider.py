"""
Groq LLM provider implementation using LangChain.
"""

import time
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.config import settings
from src.providers.base import (
    BaseProvider,
    LLMRequest,
    LLMResponse,
    ModelInfo,
    ProviderHealth,
    ProviderStatus,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    ComplexityLevel,
)


class GroqProvider(BaseProvider):
    """Groq LLM provider implementation."""

    # Groq model pricing (per 1K tokens) - Updated January 2024
    MODELS = {
        "llama-3.1-8b-instant": ModelInfo(
            model_id="llama-3.1-8b-instant",
            provider="groq",
            input_cost_per_1k=0.00005,  # $0.05 per 1M input tokens
            output_cost_per_1k=0.00008,  # $0.08 per 1M output tokens
            max_tokens=8192,
            complexity_level=ComplexityLevel.SIMPLE,
        ),
        "llama-3.3-70b-versatile": ModelInfo(
            model_id="llama-3.3-70b-versatile",
            provider="groq",
            input_cost_per_1k=0.00059,  # $0.59 per 1M input tokens
            output_cost_per_1k=0.00079,  # $0.79 per 1M output tokens
            max_tokens=32768,
            complexity_level=ComplexityLevel.MODERATE,
        ),
        "llama-3.2-90b-vision-preview": ModelInfo(
            model_id="llama-3.2-90b-vision-preview",
            provider="groq",
            input_cost_per_1k=0.0009,
            output_cost_per_1k=0.0009,
            max_tokens=8192,
            complexity_level=ComplexityLevel.COMPLEX,
        ),
        "mixtral-8x7b-32768": ModelInfo(
            model_id="mixtral-8x7b-32768",
            provider="groq",
            input_cost_per_1k=0.00024,
            output_cost_per_1k=0.00024,
            max_tokens=32768,
            complexity_level=ComplexityLevel.MODERATE,
        ),
        "gemma2-9b-it": ModelInfo(
            model_id="gemma2-9b-it",
            provider="groq",
            input_cost_per_1k=0.00020,
            output_cost_per_1k=0.00020,
            max_tokens=8192,
            complexity_level=ComplexityLevel.SIMPLE,
        ),
    }

    # Default model for each complexity level
    DEFAULT_MODELS = {
        ComplexityLevel.SIMPLE: "llama-3.1-8b-instant",
        ComplexityLevel.MODERATE: "llama-3.3-70b-versatile",
        ComplexityLevel.COMPLEX: "llama-3.3-70b-versatile",
    }

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("groq")
        self.api_key = api_key or settings.groq_api_key
        self._models = self.MODELS.copy()
        self._client: Optional[ChatGroq] = None

    def _get_client(self, model: str, temperature: float = 0.7) -> ChatGroq:
        """Get or create a ChatGroq client."""
        return ChatGroq(
            groq_api_key=self.api_key,
            model_name=model,
            temperature=temperature,
        )

    def _convert_messages(self, request: LLMRequest) -> list:
        """Convert LLMRequest messages to LangChain format."""
        lc_messages = []
        for msg in request.messages:
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
        return lc_messages

    def get_model_for_complexity(self, complexity: ComplexityLevel) -> str:
        """Get the recommended model for a complexity level."""
        return self.DEFAULT_MODELS.get(complexity, "llama-3.1-8b-instant")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from Groq."""
        # Determine model to use
        model = request.model or self.get_model_for_complexity(ComplexityLevel.MODERATE)

        if model not in self._models:
            raise InvalidRequestError("groq", f"Unknown model: {model}")

        start_time = time.time()

        try:
            # Create client for this request
            client = self._get_client(
                model=model,
                temperature=request.temperature,
            )

            # Convert messages
            lc_messages = self._convert_messages(request)

            # Invoke the model
            response = await client.ainvoke(lc_messages)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Extract token usage from response metadata
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                input_tokens = usage_metadata.get('input_tokens', 0)
                output_tokens = usage_metadata.get('output_tokens', 0)
            else:
                # Fallback: estimate tokens (rough approximation)
                input_tokens = sum(len(m.content.split()) * 1.3 for m in request.messages)
                output_tokens = len(response.content.split()) * 1.3
                input_tokens = int(input_tokens)
                output_tokens = int(output_tokens)

            total_tokens = input_tokens + output_tokens

            # Calculate cost
            cost = self.calculate_cost(model, input_tokens, output_tokens)

            # Record success
            self.record_success(latency_ms)

            return LLMResponse(
                content=response.content,
                model=model,
                provider=self.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost=cost,
                metadata=request.metadata,
            )

        except Exception as e:
            error_str = str(e).lower()
            self.record_failure(str(e))

            # Classify error type
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError("groq")
            elif "authentication" in error_str or "401" in error_str or "api key" in error_str:
                raise AuthenticationError("groq")
            elif "invalid" in error_str or "400" in error_str:
                raise InvalidRequestError("groq", str(e))
            else:
                raise ProviderError("groq", str(e), retryable=True)

    async def health_check(self) -> ProviderHealth:
        """Check Groq provider health with a simple request."""
        start_time = time.time()

        try:
            client = self._get_client(
                model="llama-3.1-8b-instant",
                temperature=0.0,
            )

            # Simple health check message
            messages = [HumanMessage(content="Hi")]
            response = await client.ainvoke(messages)

            latency_ms = (time.time() - start_time) * 1000

            self._health.status = ProviderStatus.HEALTHY
            self._health.latency_ms = latency_ms
            self._health.last_error = None

        except Exception as e:
            self._health.status = ProviderStatus.UNHEALTHY
            self._health.last_error = str(e)

        return self._health

    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available Groq models."""
        return list(self._models.values())

    def get_cheapest_model(self) -> str:
        """Get the cheapest model available."""
        return "llama-3.1-8b-instant"

    def get_fastest_model(self) -> str:
        """Get the fastest model (same as cheapest for Groq)."""
        return "llama-3.1-8b-instant"

    def get_best_model(self) -> str:
        """Get the best quality model."""
        return "llama-3.3-70b-versatile"
