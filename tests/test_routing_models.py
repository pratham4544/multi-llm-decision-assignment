"""
Tests to verify LLM routing: coding → kimi-k2, design/diagram → llama-3.3-70b.

Directly tests the select_provider node from graph.py to confirm
that keyword-based routing picks the correct model.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.providers.base import LLMRequest, Message, ComplexityLevel
from src.router.state import RouterState, WorkflowStatus


def _make_state(user_message: str) -> RouterState:
    """Build a minimal RouterState with the given user message."""
    request = LLMRequest(
        messages=[Message(role="user", content=user_message)],
        model="auto",
    )
    return RouterState(
        request=request,
        client_id="test",
        trace_id="test-trace",
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


def _run_select_provider(user_message: str) -> str:
    """Run select_provider and return the selected model name."""
    from src.router.graph import select_provider

    state = _make_state(user_message)

    # Mock health monitor and rate limiter so they don't interfere
    mock_health = MagicMock()
    mock_health.get_all_health.return_value = {}

    mock_rate_limiter = MagicMock()
    mock_rate_status = MagicMock()
    mock_rate_status.remaining = 100
    mock_rate_status.limit = 100
    mock_rate_limiter.get_status.return_value = mock_rate_status

    with patch("src.router.graph.get_health_monitor", return_value=mock_health), \
         patch("src.router.graph.get_rate_limiter", return_value=mock_rate_limiter):
        result = select_provider(state)

    return result["selected_model"]


# ── Coding tasks → moonshotai/kimi-k2-instruct-0905 ──

class TestCodingRouting:
    """Coding-related prompts must route to kimi-k2."""

    EXPECTED_MODEL = "moonshotai/kimi-k2-instruct-0905"

    def test_write_code(self):
        assert _run_select_provider("write code to sort a list") == self.EXPECTED_MODEL

    def test_write_a_function(self):
        assert _run_select_provider("write a function that reverses a string") == self.EXPECTED_MODEL

    def test_python_keyword(self):
        assert _run_select_provider("how to read a file in python") == self.EXPECTED_MODEL

    def test_javascript_keyword(self):
        assert _run_select_provider("create a todo app in javascript") == self.EXPECTED_MODEL

    def test_fix_this_code(self):
        assert _run_select_provider("fix this code it has a bug") == self.EXPECTED_MODEL

    def test_debug_keyword(self):
        assert _run_select_provider("debug this function for me") == self.EXPECTED_MODEL

    def test_code_block(self):
        assert _run_select_provider("```python\nprint('hello')\n```") == self.EXPECTED_MODEL

    def test_api_endpoint(self):
        assert _run_select_provider("create a REST api endpoint") == self.EXPECTED_MODEL

    def test_algorithm(self):
        assert _run_select_provider("implement a binary search algorithm") == self.EXPECTED_MODEL

    def test_sql_keyword(self):
        assert _run_select_provider("write a sql query to get all users") == self.EXPECTED_MODEL

    def test_react_keyword(self):
        assert _run_select_provider("build a react component") == self.EXPECTED_MODEL


# ── Design / diagram tasks → llama-3.3-70b-versatile ──

class TestDesignRouting:
    """Design/diagram/architecture prompts must route to llama-3.3-70b."""

    EXPECTED_MODEL = "llama-3.3-70b-versatile"

    def test_design_keyword(self):
        assert _run_select_provider("design a system for chat") == self.EXPECTED_MODEL

    def test_draw_diagram(self):
        assert _run_select_provider("draw a diagram of the system") == self.EXPECTED_MODEL

    def test_architecture_keyword(self):
        assert _run_select_provider("explain the architecture of microservices") == self.EXPECTED_MODEL

    def test_draw_arch_diagram(self):
        assert _run_select_provider("draw an architecture diagram") == self.EXPECTED_MODEL

    def test_architect_keyword(self):
        assert _run_select_provider("architect a solution for this problem") == self.EXPECTED_MODEL

    def test_analyze_keyword(self):
        assert _run_select_provider("analyze the pros and cons of this approach") == self.EXPECTED_MODEL

    def test_compare_keyword(self):
        assert _run_select_provider("compare REST vs GraphQL") == self.EXPECTED_MODEL

    def test_explain_in_detail(self):
        assert _run_select_provider("explain in detail how DNS works") == self.EXPECTED_MODEL

    def test_step_by_step(self):
        assert _run_select_provider("give me a step by step guide") == self.EXPECTED_MODEL

    def test_optimize_keyword(self):
        assert _run_select_provider("optimize the performance of this system") == self.EXPECTED_MODEL


# ── Simple tasks → llama-3.1-8b-instant ──

class TestSimpleRouting:
    """Simple queries must route to llama-3.1-8b-instant."""

    EXPECTED_MODEL = "llama-3.1-8b-instant"

    def test_greeting(self):
        assert _run_select_provider("hello how are you") == self.EXPECTED_MODEL

    def test_simple_question(self):
        assert _run_select_provider("what is the capital of France") == self.EXPECTED_MODEL

    def test_short_query(self):
        assert _run_select_provider("thanks") == self.EXPECTED_MODEL
