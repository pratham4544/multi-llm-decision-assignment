"""
Cache system tests.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.cache.exact_cache import ExactCache
from src.cache.semantic_cache import SemanticCache
from src.cache.cache_manager import CacheManager, CacheResult
from src.providers.base import LLMRequest, LLMResponse, Message


class TestExactCache:
    """Tests for the exact match cache."""

    def test_generate_cache_key(self):
        """Test cache key generation is deterministic."""
        cache = ExactCache(redis_client=MagicMock())
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="test-model",
            temperature=0.7,
        )
        key1 = cache._generate_cache_key(request)
        key2 = cache._generate_cache_key(request)
        assert key1 == key2

    def test_different_requests_different_keys(self):
        """Test different requests produce different keys."""
        cache = ExactCache(redis_client=MagicMock())
        request1 = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="test-model",
        )
        request2 = LLMRequest(
            messages=[Message(role="user", content="Hi")],
            model="test-model",
        )
        key1 = cache._generate_cache_key(request1)
        key2 = cache._generate_cache_key(request2)
        assert key1 != key2

    def test_get_returns_none_on_miss(self, mock_redis):
        """Test get returns None on cache miss."""
        mock_redis.get.return_value = None
        cache = ExactCache(redis_client=mock_redis)
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="test-model",
        )
        result = cache.get(request)
        assert result is None

    def test_set_stores_response(self, mock_redis):
        """Test set stores response in cache."""
        mock_redis.setex.return_value = True
        cache = ExactCache(redis_client=mock_redis)
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="test-model",
        )
        response = LLMResponse(
            content="Hi there!",
            model="test-model",
            provider="test",
            input_tokens=5,
            output_tokens=3,
            total_tokens=8,
            latency_ms=50.0,
            cost=0.0001,
        )
        cache.set(request, response)
        assert mock_redis.setex.called


class TestSemanticCache:
    """Tests for the semantic similarity cache."""

    def test_find_similar_returns_none_on_empty_cache(self):
        """Test find_similar returns None when cache is empty."""
        cache = SemanticCache(redis_client=MagicMock())
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="test-model",
        )
        result = cache.find_similar(request)
        assert result is None


class TestCacheManager:
    """Tests for the cache manager."""

    def test_get_returns_miss_on_empty_cache(self, mock_redis):
        """Test get returns cache miss when nothing is cached."""
        mock_redis.get.return_value = None
        manager = CacheManager(redis_client=mock_redis)

        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="test-model",
        )
        result = manager.get(request)
        assert result.hit is False

    def test_get_stats_returns_data(self, mock_redis):
        """Test get_stats returns statistics."""
        mock_redis.get.return_value = None
        mock_redis.hgetall.return_value = {}
        manager = CacheManager(redis_client=mock_redis)
        stats = manager.get_stats()
        assert "enabled" in stats
        assert "ttl" in stats
