"""
API route handlers for the LLM router.
"""

import time
from datetime import datetime
from typing import Any, Dict
import uuid
import structlog

from fastapi import APIRouter, Depends, HTTPException, Response, status

from src.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChatMessage,
    UsageInfo,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    CostAnalysisResponse,
)
from src.api.dependencies import (
    RequestContext,
    get_request_context,
    rate_limit_headers,
    get_cache_service,
    get_health_service,
    get_rate_limit_service,
)
from src.providers.base import LLMRequest, Message
from src.providers.factory import ProviderFactory
from src.providers.health_monitor import get_health_monitor
from src.router.graph import route_request
from src.router.state import WorkflowStatus
from src.cache.cache_manager import get_cache_manager
from src.resilience.circuit_breaker import get_circuit_breaker_manager
from src.resilience.fallback_manager import get_fallback_manager

logger = structlog.get_logger()

router = APIRouter()


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Create chat completion",
    description="Generate a chat completion using intelligent routing across LLM providers.",
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    response: Response,
    ctx: RequestContext = Depends(get_request_context),
) -> ChatCompletionResponse:
    """
    Create a chat completion with intelligent routing.

    This endpoint:
    1. Analyzes request complexity
    2. Checks cache for existing response
    3. Routes to optimal provider/model
    4. Handles failures with fallback
    5. Caches the response
    """
    start_time = time.time()

    logger.info(
        "chat_completion_request",
        trace_id=ctx.trace_id,
        client_id=ctx.client_id,
        tier=ctx.tier,
        message_count=len(request.messages),
        model=request.model,
    )

    try:
        # Convert to internal request format
        llm_request = LLMRequest(
            messages=[
                Message(role=m.role, content=m.content)
                for m in request.messages
            ],
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            metadata={
                "client_id": ctx.client_id,
                "tier": ctx.tier,
                **request.metadata,
            },
        )

        # Route the request through the workflow
        final_state = await route_request(
            request=llm_request,
            client_id=ctx.client_id,
            trace_id=ctx.trace_id,
        )

        # Check for success
        if final_state.get("status") == WorkflowStatus.FAILED or not final_state.get("response"):
            last_error = final_state.get("last_error", "Unknown error")
            logger.error(
                "chat_completion_failed",
                trace_id=ctx.trace_id,
                error=last_error,
                attempts=final_state.get("attempt_count", 0),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": f"All providers failed: {last_error}",
                        "type": "provider_error",
                        "code": "all_providers_failed",
                    }
                }
            )

        llm_response = final_state["response"]
        total_latency = (time.time() - start_time) * 1000

        # Build response
        completion_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(datetime.utcnow().timestamp()),
            model=llm_response.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=llm_response.content
                    ),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=llm_response.input_tokens,
                completion_tokens=llm_response.output_tokens,
                total_tokens=llm_response.total_tokens,
            ),
            provider=llm_response.provider,
            cached=llm_response.cached,
            cache_type=llm_response.cache_type,
            latency_ms=round(total_latency, 2),
            cost=llm_response.cost,
            trace_id=ctx.trace_id,
        )

        # Add rate limit headers
        response.headers.update(rate_limit_headers(ctx.rate_limit))
        response.headers["X-Trace-ID"] = ctx.trace_id

        logger.info(
            "chat_completion_success",
            trace_id=ctx.trace_id,
            provider=llm_response.provider,
            model=llm_response.model,
            cached=llm_response.cached,
            latency_ms=round(total_latency, 2),
            cost=llm_response.cost,
        )

        return completion_response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "chat_completion_error",
            trace_id=ctx.trace_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_server_error",
                }
            }
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health of the service and its dependencies.",
)
async def health_check() -> HealthResponse:
    """Get service health status."""
    import redis
    from src.config import settings

    # Check Redis
    redis_status = {"status": "unknown"}
    try:
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
        )
        r.ping()
        redis_status = {"status": "healthy", "connected": True}
    except Exception as e:
        redis_status = {"status": "unhealthy", "error": str(e)}

    # Check providers
    health_monitor = get_health_monitor()
    provider_status = health_monitor.get_status_summary()

    overall_status = "healthy"
    if redis_status.get("status") != "healthy":
        overall_status = "degraded"
    if provider_status.get("unhealthy", 0) > 0:
        overall_status = "degraded"
    if provider_status.get("healthy", 0) == 0 and provider_status.get("unknown", 0) == 0:
        overall_status = "unhealthy"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.utcnow(),
        providers=provider_status.get("providers", {}),
        redis=redis_status,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get metrics",
    description="Get current system metrics.",
)
async def get_metrics() -> MetricsResponse:
    """Get current metrics."""
    # Cache metrics
    cache_manager = get_cache_manager()
    cache_stats = cache_manager.get_stats()

    # Provider metrics
    health_monitor = get_health_monitor()
    provider_stats = health_monitor.get_status_summary()

    # Circuit breaker metrics
    cb_manager = get_circuit_breaker_manager()
    cb_stats = {
        name: {
            "state": stats.state.value,
            "failure_count": stats.failure_count,
            "success_count": stats.success_count,
            "failure_rate": stats.failure_rate,
        }
        for name, stats in cb_manager.get_all_stats().items()
    }

    # Fallback metrics
    fallback_manager = get_fallback_manager()
    fallback_stats = fallback_manager.get_metrics()

    # Rate limit metrics
    rate_limiter = get_rate_limit_service()
    rate_limit_stats = {
        "active_clients": len(rate_limiter.get_all_limits()),
    }

    return MetricsResponse(
        requests={
            "total": fallback_stats.get("total_attempts", 0),
            "fallback_count": fallback_stats.get("fallback_count", 0),
            "fallback_rate": fallback_stats.get("fallback_rate", 0),
        },
        cache=cache_stats,
        providers={
            "summary": provider_stats,
            "circuit_breakers": cb_stats,
        },
        rate_limits=rate_limit_stats,
        costs=cache_manager.get_savings_estimate(),
    )


@router.get(
    "/cost-analysis",
    response_model=CostAnalysisResponse,
    summary="Get cost analysis",
    description="Get cost analysis and optimization metrics.",
)
async def get_cost_analysis() -> CostAnalysisResponse:
    """Get cost analysis."""
    cache_manager = get_cache_manager()
    savings = cache_manager.get_savings_estimate()

    # In a real implementation, these would come from a metrics store
    return CostAnalysisResponse(
        total_cost_usd=0.0,  # Would be tracked in real implementation
        cost_by_provider={"groq": 0.0},
        cost_by_model={"llama-3.1-8b-instant": 0.0},
        cache_savings_usd=savings.get("estimated_cost_saved_usd", 0.0),
        requests_processed=savings.get("cache_hits", 0),
        average_cost_per_request=0.0,
        period="session",
    )


@router.post(
    "/cache/clear",
    summary="Clear cache",
    description="Clear all cached responses.",
)
async def clear_cache() -> Dict[str, Any]:
    """Clear all caches."""
    cache_manager = get_cache_manager()
    cleared = cache_manager.clear()

    logger.info("cache_cleared_via_api", cleared=cleared)

    return {
        "status": "success",
        "cleared": cleared,
    }


@router.get(
    "/providers",
    summary="List providers",
    description="Get list of available providers and their models.",
)
async def list_providers() -> Dict[str, Any]:
    """List available providers and models."""
    providers = ProviderFactory.get_all_providers()
    health_monitor = get_health_monitor()

    result = {}
    for name, provider in providers.items():
        health = health_monitor.get_health(name)
        models = provider.get_available_models()

        result[name] = {
            "status": health.status.value if health else "unknown",
            "models": [
                {
                    "id": m.model_id,
                    "complexity": m.complexity_level.value,
                    "max_tokens": m.max_tokens,
                    "input_cost_per_1k": m.input_cost_per_1k,
                    "output_cost_per_1k": m.output_cost_per_1k,
                }
                for m in models
            ],
        }

    return result
