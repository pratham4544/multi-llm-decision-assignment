"""
Pytest configuration and fixtures.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.app import app
from src.providers.base import LLMRequest, LLMResponse, Message
from src.cache.cache_manager import CacheManager


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_request():
    """Create a sample LLM request."""
    return LLMRequest(
        messages=[Message(role="user", content="Hello, how are you?")],
        model="auto",
        temperature=0.7,
        max_tokens=100,
    )


@pytest.fixture
def sample_response():
    """Create a sample LLM response."""
    return LLMResponse(
        content="I'm doing well, thank you!",
        model="llama-3.1-8b-instant",
        provider="groq",
        input_tokens=10,
        output_tokens=8,
        total_tokens=18,
        latency_ms=150.5,
        cost=0.00001,
    )


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = MagicMock()
    redis_mock.get = MagicMock(return_value=None)
    redis_mock.set = MagicMock(return_value=True)
    redis_mock.setex = MagicMock(return_value=True)
    redis_mock.delete = MagicMock(return_value=True)
    redis_mock.ping = MagicMock(return_value=True)
    redis_mock.hincrby = MagicMock(return_value=1)
    return redis_mock


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.name = "mock-provider"
    provider.generate = AsyncMock(return_value=LLMResponse(
        content="Mock response",
        model="mock-model",
        provider="mock-provider",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        latency_ms=100.0,
        cost=0.0001,
    ))
    provider.health_check = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
