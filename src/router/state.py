"""
LangGraph state definitions for the routing workflow.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from src.providers.base import LLMRequest, LLMResponse, ComplexityLevel
from src.router.analyzer import RequestAnalysis


class WorkflowStatus(Enum):
    """Status of the routing workflow."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    CACHE_CHECK = "cache_check"
    ROUTING = "routing"
    EXECUTING = "executing"
    FALLBACK = "fallback"
    COMPLETED = "completed"
    FAILED = "failed"


class RouterState(TypedDict, total=False):
    """
    State definition for the LangGraph routing workflow.

    This state flows through all nodes in the graph and
    accumulates information as the request is processed.
    """
    # Input
    request: LLMRequest
    client_id: str
    trace_id: str

    # Analysis
    analysis: Optional[RequestAnalysis]
    complexity: Optional[ComplexityLevel]
    token_count: int
    question_type: str

    # Cache
    cache_hit: bool
    cache_type: Optional[str]
    cached_response: Optional[LLMResponse]

    # Routing
    selected_provider: Optional[str]
    selected_model: Optional[str]
    fallback_providers: List[str]
    fallback_index: int

    # Execution
    current_provider: Optional[str]
    attempt_count: int
    last_error: Optional[str]

    # Output
    response: Optional[LLMResponse]
    status: WorkflowStatus
    total_latency_ms: float

    # Metrics
    start_time: datetime
    provider_latencies: Dict[str, float]
    errors: List[Dict[str, Any]]


@dataclass
class RouterContext:
    """
    Additional context for routing decisions.
    Not part of the graph state, but used by decision engine.
    """
    # Provider health status
    healthy_providers: List[str] = field(default_factory=list)
    degraded_providers: List[str] = field(default_factory=list)
    unhealthy_providers: List[str] = field(default_factory=list)

    # Cost constraints
    max_cost: Optional[float] = None
    cost_priority: float = 0.5  # 0 = quality, 1 = cost

    # Latency constraints
    max_latency_ms: Optional[float] = None
    latency_priority: float = 0.3

    # Quality requirements
    min_quality_score: float = 0.7

    # Client tier (affects rate limits, model access)
    client_tier: str = "standard"  # standard, premium, enterprise

    # Question type for specialized routing
    question_type: Optional[str] = None

    # Rate pressure indicator
    high_request_rate: bool = False


def create_initial_state(
    request: LLMRequest,
    client_id: str = "default",
    trace_id: Optional[str] = None
) -> RouterState:
    """
    Create initial router state for a new request.

    Args:
        request: The LLM request to process
        client_id: Client identifier for tracking
        trace_id: Optional trace ID (generated if not provided)

    Returns:
        Initial RouterState
    """
    import uuid

    if not trace_id:
        trace_id = str(uuid.uuid4())

    return RouterState(
        request=request,
        client_id=client_id,
        trace_id=trace_id,
        analysis=None,
        complexity=None,
        token_count=0,
        question_type="simple",
        cache_hit=False,
        cache_type=None,
        cached_response=None,
        selected_provider=None,
        selected_model=None,
        fallback_providers=[],
        fallback_index=0,
        current_provider=None,
        attempt_count=0,
        last_error=None,
        response=None,
        status=WorkflowStatus.PENDING,
        total_latency_ms=0.0,
        start_time=datetime.utcnow(),
        provider_latencies={},
        errors=[],
    )


def get_state_summary(state: RouterState) -> Dict[str, Any]:
    """Get a summary of the current state for logging/tracing."""
    return {
        "trace_id": state.get("trace_id"),
        "client_id": state.get("client_id"),
        "status": state.get("status", WorkflowStatus.PENDING).value,
        "complexity": state.get("complexity").value if state.get("complexity") else None,
        "cache_hit": state.get("cache_hit", False),
        "cache_type": state.get("cache_type"),
        "selected_provider": state.get("selected_provider"),
        "selected_model": state.get("selected_model"),
        "attempt_count": state.get("attempt_count", 0),
        "total_latency_ms": state.get("total_latency_ms", 0),
        "has_response": state.get("response") is not None,
        "last_error": state.get("last_error"),
    }
