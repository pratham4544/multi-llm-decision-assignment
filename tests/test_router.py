"""
Router logic tests.
"""

import pytest
from src.router.analyzer import RequestAnalyzer, RequestAnalysis
from src.router.decision_engine import DecisionEngine, RoutingDecision
from src.providers.base import LLMRequest, Message, ComplexityLevel


class TestRequestAnalyzer:
    """Tests for request complexity analysis."""

    def test_simple_request_classification(self):
        """Test simple requests are classified correctly."""
        analyzer = RequestAnalyzer()
        request = LLMRequest(
            messages=[Message(role="user", content="What is 2+2?")],
            model="auto",
        )
        analysis = analyzer.analyze(request)
        assert analysis.complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]
        assert analysis.token_count > 0

    def test_complex_request_classification(self):
        """Test complex requests are classified correctly."""
        analyzer = RequestAnalyzer()
        request = LLMRequest(
            messages=[Message(
                role="user",
                content="""Write a comprehensive Python program that implements a distributed
                microservices architecture with the following detailed requirements. First, you need
                to implement service discovery using Consul with health checking and automatic
                registration. Second, implement load balancing with both round-robin and weighted
                strategies that can dynamically adjust based on server health. Third, add a circuit
                breaker pattern for fault tolerance with configurable thresholds. Fourth, integrate
                distributed tracing with OpenTelemetry for debugging and monitoring. Finally, set up
                gRPC communication between all services with proper authentication.
                Include detailed documentation, comprehensive unit tests, and example usage."""
            )],
            model="auto",
        )
        analysis = analyzer.analyze(request)
        # This should be detected as complex due to length and content
        assert analysis.complexity in [ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX]

    def test_code_detection(self):
        """Test code content is detected."""
        analyzer = RequestAnalyzer()
        request = LLMRequest(
            messages=[Message(role="user", content="def hello(): print('world')")],
            model="auto",
        )
        analysis = analyzer.analyze(request)
        assert analysis.has_code is True

    def test_math_detection(self):
        """Test math content is detected."""
        analyzer = RequestAnalyzer()
        request = LLMRequest(
            messages=[Message(role="user", content="Calculate 5 + 3 * 2")],
            model="auto",
        )
        analysis = analyzer.analyze(request)
        assert analysis.has_math is True

    def test_analysis_returns_recommendations(self):
        """Test analysis includes model recommendations."""
        analyzer = RequestAnalyzer()
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="auto",
        )
        analysis = analyzer.analyze(request)
        assert analysis.recommended_model is not None
        assert analysis.recommended_provider is not None


class TestDecisionEngine:
    """Tests for routing decision engine."""

    def test_makes_routing_decision(self):
        """Test engine makes valid routing decisions."""
        engine = DecisionEngine()
        decision = engine.decide(
            complexity=ComplexityLevel.SIMPLE,
            token_count=50
        )
        assert decision is not None
        assert decision.model is not None
        assert decision.provider is not None

    def test_complex_requests_route_appropriately(self):
        """Test complex requests are routed appropriately."""
        engine = DecisionEngine()
        decision = engine.decide(
            complexity=ComplexityLevel.COMPLEX,
            token_count=500
        )
        assert decision is not None
        # Should route to a model (implementation may vary)
        assert decision.model is not None

    def test_decision_includes_reason(self):
        """Test decision includes reason."""
        engine = DecisionEngine()
        decision = engine.decide(
            complexity=ComplexityLevel.MODERATE,
            token_count=100
        )
        assert decision.reason is not None
        assert len(decision.reason) > 0

    def test_fallback_chain_provided(self):
        """Test decision includes fallback chain."""
        engine = DecisionEngine()
        decision = engine.decide(
            complexity=ComplexityLevel.SIMPLE,
            token_count=50
        )
        # Fallback chain should be a list (may be empty if only one option)
        assert isinstance(decision.fallback_chain, list)
