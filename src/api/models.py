"""
Pydantic models for API requests and responses.
OpenAI-compatible chat completion format.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message in a conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """
    Chat completion request (OpenAI-compatible format).
    """
    messages: List[ChatMessage] = Field(
        ...,
        description="List of messages in the conversation"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use (optional, auto-selected if not provided)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=32768,
        description="Maximum tokens to generate"
    )
    stream: bool = Field(
        default=False,
        description="Stream the response (not yet supported)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "temperature": 0.7
            }
        }


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """A completion choice."""
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """
    Chat completion response (OpenAI-compatible format).
    """
    id: str = Field(description="Unique response ID")
    object: str = Field(default="chat.completion")
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used")
    choices: List[Choice]
    usage: UsageInfo

    # Extended fields
    provider: str = Field(description="LLM provider used")
    cached: bool = Field(default=False, description="Whether response was cached")
    cache_type: Optional[str] = Field(default=None, description="Cache type if cached")
    latency_ms: float = Field(description="Response latency in milliseconds")
    cost: float = Field(description="Request cost in USD")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for debugging")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-abc123",
                "object": "chat.completion",
                "created": 1677858242,
                "model": "llama-3.1-8b-instant",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The capital of France is Paris."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 10,
                    "total_tokens": 25
                },
                "provider": "groq",
                "cached": False,
                "latency_ms": 150.5,
                "cost": 0.000001,
                "trace_id": "abc-123-def"
            }
        }


class ErrorDetail(BaseModel):
    """Error detail information."""
    message: str
    type: str
    code: Optional[str] = None
    param: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response format."""
    error: ErrorDetail

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "message": "Rate limit exceeded. Please try again in 30 seconds.",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded"
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime
    providers: Dict[str, Dict[str, Any]]
    redis: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Metrics endpoint response."""
    requests: Dict[str, Any]
    cache: Dict[str, Any]
    providers: Dict[str, Any]
    rate_limits: Dict[str, Any]
    costs: Dict[str, Any]


class CostAnalysisResponse(BaseModel):
    """Cost analysis response."""
    total_cost_usd: float
    cost_by_provider: Dict[str, float]
    cost_by_model: Dict[str, float]
    cache_savings_usd: float
    requests_processed: int
    average_cost_per_request: float
    period: str
