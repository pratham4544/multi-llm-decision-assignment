"""
API endpoint tests.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.api.app import app
from src.providers.base import LLMResponse
from src.router.state import WorkflowStatus


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, test_client):
        """Test health endpoint returns healthy status."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "timestamp" in data

    def test_health_includes_providers(self, test_client):
        """Test health endpoint includes provider status."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_returns_data(self, test_client):
        """Test metrics endpoint returns metric data."""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests" in data
        assert "cache" in data
        assert "providers" in data


class TestChatCompletionsEndpoint:
    """Tests for the /v1/chat/completions endpoint."""

    def test_chat_completions_requires_messages(self, test_client):
        """Test chat completions requires messages field."""
        response = test_client.post(
            "/v1/chat/completions",
            json={"model": "auto"},
            headers={"X-Client-ID": "test-client"}
        )
        assert response.status_code == 422

    def test_chat_completions_validates_message_format(self, test_client):
        """Test chat completions validates message format."""
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"invalid": "format"}],
                "model": "auto"
            },
            headers={"X-Client-ID": "test-client"}
        )
        assert response.status_code == 422

    @patch("src.api.routes.route_request")
    def test_chat_completions_success(self, mock_route, test_client):
        """Test successful chat completion request."""
        mock_response = LLMResponse(
            content="Hello!",
            model="llama-3.1-8b-instant",
            provider="groq",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_ms=100.0,
            cost=0.0001,
        )
        # route_request returns a state dict
        mock_route.return_value = {
            "status": WorkflowStatus.COMPLETED,
            "response": mock_response,
            "cached": False,
            "cache_type": None,
        }

        response = test_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "model": "auto"
            },
            headers={"X-Client-ID": "test-client"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["provider"] == "groq"


class TestProvidersEndpoint:
    """Tests for the /providers endpoint."""

    def test_list_providers(self, test_client):
        """Test listing available providers."""
        response = test_client.get("/providers")
        assert response.status_code == 200
        data = response.json()
        # The endpoint returns providers as keys
        assert "groq" in data or "providers" in data
