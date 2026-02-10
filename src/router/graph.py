"""
LangGraph routing workflow implementation.
Defines the state machine for processing LLM requests.
"""

import time
from datetime import datetime
from typing import Any, Dict, Literal, Optional
import structlog

from langgraph.graph import StateGraph, END

from src.cache.cache_manager import get_cache_manager
from src.providers.base import LLMRequest, LLMResponse, ProviderError, ComplexityLevel
from src.providers.factory import ProviderFactory
from src.providers.health_monitor import get_health_monitor
from src.router.analyzer import get_analyzer, RequestAnalysis
from src.rate_limit.token_bucket import get_rate_limiter
from src.router.decision_engine import get_decision_engine
from src.router.state import RouterState, RouterContext, WorkflowStatus, create_initial_state

logger = structlog.get_logger()


# Node functions for the routing graph

def analyze_request(state: RouterState) -> RouterState:
    """Analyze the incoming request for complexity and routing."""
    logger.debug("node_analyze_request", trace_id=state.get("trace_id"))

    state["status"] = WorkflowStatus.ANALYZING

    request = state["request"]
    analyzer = get_analyzer()

    analysis = analyzer.analyze(request)

    state["analysis"] = analysis
    state["complexity"] = analysis.complexity
    state["token_count"] = analysis.token_count
    state["question_type"] = analysis.question_type

    logger.info(
        "request_analyzed",
        trace_id=state.get("trace_id"),
        complexity=analysis.complexity.value,
        token_count=analysis.token_count,
        question_type=analysis.question_type,
    )

    return state


def check_cache(state: RouterState) -> RouterState:
    """Check for cached response."""
    logger.debug("node_check_cache", trace_id=state.get("trace_id"))

    state["status"] = WorkflowStatus.CACHE_CHECK

    request = state["request"]
    cache_manager = get_cache_manager()

    cache_result = cache_manager.get(request)

    if cache_result.hit:
        state["cache_hit"] = True
        state["cache_type"] = cache_result.cache_type
        state["cached_response"] = cache_result.response
        state["response"] = cache_result.response
        state["status"] = WorkflowStatus.COMPLETED

        logger.info(
            "cache_hit",
            trace_id=state.get("trace_id"),
            cache_type=cache_result.cache_type,
            similarity=cache_result.similarity_score,
        )
    else:
        state["cache_hit"] = False
        logger.debug("cache_miss", trace_id=state.get("trace_id"))

    return state


def select_provider(state: RouterState) -> RouterState:
    """Select provider/model using direct keyword matching on the user message."""
    logger.debug("node_select_provider", trace_id=state.get("trace_id"))

    state["status"] = WorkflowStatus.ROUTING

    # Get the raw user message text
    request = state["request"]
    user_text = (request.last_user_message or request.prompt_text or "").lower()
    # Pad with spaces for safe word-boundary matching
    padded = f" {user_text} "

    # --- CODING keywords → mixtral-8x7b-32768 ---
    # Use spaces around short words to avoid substring false matches
    # e.g. " api " won't match "capital", " class " won't match "classical"
    CODING_WORDS = [
        'write a function', 'write code', 'fix this code', 'code to',
        'write a program', 'write a script', 'coding challenge',
        ' code ', ' function ', ' algorithm ', ' python ', ' javascript ',
        ' typescript ', ' java ', ' c++ ', ' rust ', ' golang ',
        ' html ', ' css ', ' react ', ' angular ', ' vue ',
        ' implement ', ' program ', ' script ', ' debug ', ' refactor ',
        ' compile ', ' api ', ' endpoint ', ' class ', ' method ',
        ' variable ', ' loop ', ' array ', ' sort ', ' regex ',
        ' sql ', ' database ', ' leetcode ', ' hackerrank ',
        'binary search', 'linked list', 'hash map', 'unit test',
        'def ', 'print(', 'import ', '```',
    ]

    # --- COMPLEX keywords → llama-3.3-70b-versatile ---
    COMPLEX_WORDS = [
        'analyze', 'compare', 'evaluate', 'synthesize', 'critique',
        'explain in detail', 'step by step', 'comprehensive', 'in-depth',
        'research', 'investigate', 'deep dive', 'pros and cons',
        'design pattern', 'architect', 'optimize', 'trade-off',
        'implications', 'essay', 'dissertation', 'thesis',
        'philosophy', 'economics',
    ]

    # --- Direct matching (check in priority order) ---
    selected_model = None
    reason = ""

    # Priority 1: Check for high latency → gemma2
    try:
        health_monitor = get_health_monitor()
        all_health = health_monitor.get_all_health()
        for pname, health in all_health.items():
            if health.latency_ms is not None and health.latency_ms > 1000:
                selected_model = "gemma2-9b-it"
                reason = f"Latency fallback: {pname} showing high latency"
                break
    except Exception:
        pass

    # Priority 2: Check for high request rate → gemma2
    if not selected_model:
        try:
            rate_limiter = get_rate_limiter()
            client_id = state.get("client_id", "default")
            rate_status = rate_limiter.get_status(client_id)
            if rate_status.remaining < rate_status.limit * 0.2:
                selected_model = "gemma2-9b-it"
                reason = "High request rate, using lightweight model"
        except Exception:
            pass

    # Priority 3: Coding keywords → mixtral
    # Check against both raw text and space-padded text
    if not selected_model:
        for word in CODING_WORDS:
            if word in padded or word in user_text:
                selected_model = "mixtral-8x7b-32768"
                reason = f"Coding task detected (matched: '{word.strip()}')"
                break

    # Priority 4: Complex keywords → 70b
    if not selected_model:
        for word in COMPLEX_WORDS:
            if word in user_text:
                selected_model = "llama-3.3-70b-versatile"
                reason = f"Complex task detected (matched: '{word}')"
                break

    # Priority 5: Default → simple 8b
    if not selected_model:
        selected_model = "llama-3.1-8b-instant"
        reason = "Simple query, using fast model"

    # All models are on groq
    selected_provider = "groq"

    # Build fallback chain (all other models)
    all_models = [
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
    ]
    fallbacks = [
        f"groq:{m}" for m in all_models if m != selected_model
    ]

    state["selected_provider"] = selected_provider
    state["selected_model"] = selected_model
    state["fallback_providers"] = fallbacks
    state["current_provider"] = selected_provider

    logger.info(
        "provider_selected",
        trace_id=state.get("trace_id"),
        provider=selected_provider,
        model=selected_model,
        reason=reason,
        user_text_preview=user_text[:80],
    )

    return state


async def execute_llm(state: RouterState) -> RouterState:
    """Execute the LLM request with the selected provider."""
    logger.debug("node_execute_llm", trace_id=state.get("trace_id"))

    state["status"] = WorkflowStatus.EXECUTING
    state["attempt_count"] = state.get("attempt_count", 0) + 1

    provider_name = state.get("current_provider", "groq")
    model = state.get("selected_model", "llama-3.1-8b-instant")
    request = state["request"]

    # Update request with selected model
    request.model = model

    provider = ProviderFactory.get_provider(provider_name)

    if not provider:
        state["last_error"] = f"Provider {provider_name} not available"
        logger.error(
            "provider_not_available",
            trace_id=state.get("trace_id"),
            provider=provider_name,
        )
        return state

    start_time = time.time()

    try:
        response = await provider.generate(request)

        latency_ms = (time.time() - start_time) * 1000
        response.trace_id = state.get("trace_id")

        state["response"] = response
        state["status"] = WorkflowStatus.COMPLETED
        state["total_latency_ms"] = latency_ms
        state["provider_latencies"][provider_name] = latency_ms

        # Cache the response
        cache_manager = get_cache_manager()
        cache_manager.set(request, response)

        logger.info(
            "llm_execution_success",
            trace_id=state.get("trace_id"),
            provider=provider_name,
            model=model,
            latency_ms=round(latency_ms, 2),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
        )

    except ProviderError as e:
        latency_ms = (time.time() - start_time) * 1000
        state["last_error"] = str(e)
        state["provider_latencies"][provider_name] = latency_ms
        state["errors"].append({
            "provider": provider_name,
            "model": model,
            "error": str(e),
            "retryable": e.retryable,
            "timestamp": datetime.utcnow().isoformat(),
        })

        logger.warning(
            "llm_execution_failed",
            trace_id=state.get("trace_id"),
            provider=provider_name,
            model=model,
            error=str(e),
            retryable=e.retryable,
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        state["last_error"] = str(e)
        state["errors"].append({
            "provider": provider_name,
            "model": model,
            "error": str(e),
            "retryable": True,
            "timestamp": datetime.utcnow().isoformat(),
        })

        logger.error(
            "llm_execution_error",
            trace_id=state.get("trace_id"),
            provider=provider_name,
            error=str(e),
        )

    return state


def try_fallback(state: RouterState) -> RouterState:
    """Try fallback provider after failure."""
    logger.debug("node_try_fallback", trace_id=state.get("trace_id"))

    state["status"] = WorkflowStatus.FALLBACK

    fallback_list = state.get("fallback_providers", [])
    fallback_index = state.get("fallback_index", 0)

    if fallback_index < len(fallback_list):
        fallback = fallback_list[fallback_index]
        provider, model = fallback.split(":")

        state["current_provider"] = provider
        state["selected_model"] = model
        state["fallback_index"] = fallback_index + 1

        logger.info(
            "trying_fallback",
            trace_id=state.get("trace_id"),
            provider=provider,
            model=model,
            attempt=fallback_index + 1,
        )
    else:
        state["status"] = WorkflowStatus.FAILED
        logger.error(
            "all_fallbacks_exhausted",
            trace_id=state.get("trace_id"),
            attempts=state.get("attempt_count", 0),
        )

    return state


# Conditional edge functions

def should_skip_llm(state: RouterState) -> Literal["skip", "continue"]:
    """Check if we should skip LLM execution (cache hit)."""
    if state.get("cache_hit"):
        return "skip"
    return "continue"


def check_execution_result(state: RouterState) -> Literal["success", "retry", "fail"]:
    """Check result of LLM execution."""
    if state.get("response") is not None:
        return "success"

    # Check if we can retry
    attempt_count = state.get("attempt_count", 0)
    fallback_list = state.get("fallback_providers", [])
    fallback_index = state.get("fallback_index", 0)

    # Max 3 attempts total
    if attempt_count >= 3:
        return "fail"

    # Check if fallbacks available
    if fallback_index < len(fallback_list):
        return "retry"

    return "fail"


def build_router_graph() -> StateGraph:
    """
    Build the LangGraph routing workflow.

    Flow:
    1. analyze_request -> check_cache
    2. check_cache -> (cache hit) -> END
    3. check_cache -> (cache miss) -> select_provider
    4. select_provider -> execute_llm
    5. execute_llm -> (success) -> END
    6. execute_llm -> (failure) -> try_fallback
    7. try_fallback -> execute_llm (retry)
    8. try_fallback -> (no fallbacks) -> END (failed)
    """
    graph = StateGraph(RouterState)

    # Add nodes
    graph.add_node("analyze_request", analyze_request)
    graph.add_node("check_cache", check_cache)
    graph.add_node("select_provider", select_provider)
    graph.add_node("execute_llm", execute_llm)
    graph.add_node("try_fallback", try_fallback)

    # Set entry point
    graph.set_entry_point("analyze_request")

    # Add edges
    graph.add_edge("analyze_request", "check_cache")

    # Conditional: cache hit or miss
    graph.add_conditional_edges(
        "check_cache",
        should_skip_llm,
        {
            "skip": END,
            "continue": "select_provider",
        }
    )

    graph.add_edge("select_provider", "execute_llm")

    # Conditional: execution result
    graph.add_conditional_edges(
        "execute_llm",
        check_execution_result,
        {
            "success": END,
            "retry": "try_fallback",
            "fail": END,
        }
    )

    # Fallback loops back to execute
    graph.add_edge("try_fallback", "execute_llm")

    return graph


# Compiled graph instance
_router_graph = None


def get_router_graph():
    """Get the compiled router graph."""
    global _router_graph
    if _router_graph is None:
        graph = build_router_graph()
        _router_graph = graph.compile()
    return _router_graph


async def route_request(
    request: LLMRequest,
    client_id: str = "default",
    trace_id: Optional[str] = None
) -> RouterState:
    """
    Route and process an LLM request through the workflow.

    Args:
        request: The LLM request to process
        client_id: Client identifier
        trace_id: Optional trace ID

    Returns:
        Final RouterState with response or error
    """
    # Create initial state
    initial_state = create_initial_state(
        request=request,
        client_id=client_id,
        trace_id=trace_id
    )

    # Get compiled graph
    graph = get_router_graph()

    # Execute the graph
    final_state = await graph.ainvoke(initial_state)

    return final_state
