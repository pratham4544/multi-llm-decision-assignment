"""
Semantic similarity caching using embeddings.
Uses cosine similarity to find similar cached responses.
"""

import hashlib
import json
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import redis
import structlog

from src.config import settings
from src.providers.base import LLMRequest, LLMResponse

logger = structlog.get_logger()


class FakeEmbeddings:
    """
    Fake embeddings for testing and development.
    Generates deterministic embeddings based on text hash.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        """Generate a fake embedding from text."""
        # Create deterministic hash-based embedding
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Convert hash to floats
        embedding = []
        for i in range(0, min(len(text_hash), self.dimension * 2), 2):
            byte_val = int(text_hash[i:i+2], 16)
            # Normalize to [-1, 1]
            normalized = (byte_val / 127.5) - 1.0
            embedding.append(normalized)

        # Pad or truncate to dimension
        while len(embedding) < self.dimension:
            embedding.append(0.0)

        embedding = embedding[:self.dimension]

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [e / norm for e in embedding]

        return embedding


class SemanticCache:
    """Semantic similarity cache using embeddings and Redis."""

    CACHE_PREFIX = "llm:cache:semantic:"
    EMBEDDING_PREFIX = "llm:cache:semantic:emb:"
    INDEX_KEY = "llm:cache:semantic:index"
    STATS_KEY = "llm:cache:semantic:stats"

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        ttl: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_fake_embeddings: bool = True
    ):
        """
        Initialize semantic cache.

        Args:
            redis_client: Redis client instance
            ttl: Cache TTL in seconds
            similarity_threshold: Minimum similarity score (0-1)
            use_fake_embeddings: Use fake embeddings for testing
        """
        self.ttl = ttl or settings.cache_ttl
        self.similarity_threshold = (
            similarity_threshold or settings.semantic_similarity_threshold
        )

        if redis_client:
            self.redis = redis_client
        else:
            self.redis = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password or None,
                decode_responses=True
            )

        # Use fake embeddings for development/testing
        if use_fake_embeddings or settings.use_fake_embeddings:
            self.embedder = FakeEmbeddings(dimension=384)
        else:
            # In production, you'd use a real embedding model
            # from sentence_transformers import SentenceTransformer
            # self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedder = FakeEmbeddings(dimension=384)

        self._hits = 0
        self._misses = 0

    def _get_prompt_text(self, request: LLMRequest) -> str:
        """Extract text to embed from request."""
        # Use the last user message or full prompt
        return request.last_user_message or request.prompt_text

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.embedder.embed(text)

    def _cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _get_cache_id(self) -> str:
        """Generate a unique cache entry ID."""
        import uuid
        return str(uuid.uuid4())[:12]

    def find_similar(
        self,
        request: LLMRequest
    ) -> Optional[Tuple[LLMResponse, float]]:
        """
        Find a similar cached response.

        Args:
            request: LLM request to find similar cache for

        Returns:
            Tuple of (LLMResponse, similarity_score) or None
        """
        prompt_text = self._get_prompt_text(request)
        query_embedding = self._generate_embedding(prompt_text)

        try:
            # Get all cached embeddings
            cached_ids = self.redis.smembers(self.INDEX_KEY)

            best_match = None
            best_score = 0.0

            for cache_id in cached_ids:
                # Get embedding
                emb_key = f"{self.EMBEDDING_PREFIX}{cache_id}"
                emb_data = self.redis.get(emb_key)

                if not emb_data:
                    continue

                cached_embedding = json.loads(emb_data)
                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                if similarity > best_score and similarity >= self.similarity_threshold:
                    best_score = similarity
                    best_match = cache_id

            if best_match:
                # Retrieve cached response
                cache_key = f"{self.CACHE_PREFIX}{best_match}"
                cached_data = self.redis.get(cache_key)

                if cached_data:
                    self._hits += 1
                    self._update_stats("hits")

                    data = json.loads(cached_data)

                    logger.debug(
                        "cache_hit",
                        cache_type="semantic",
                        similarity=round(best_score, 4),
                        cache_id=best_match
                    )

                    response = LLMResponse(
                        content=data["content"],
                        model=data["model"],
                        provider=data["provider"],
                        input_tokens=data["input_tokens"],
                        output_tokens=data["output_tokens"],
                        total_tokens=data["total_tokens"],
                        latency_ms=0.0,
                        cost=0.0,
                        cached=True,
                        cache_type="semantic",
                        created_at=datetime.fromisoformat(data["created_at"]),
                        metadata={
                            **data.get("metadata", {}),
                            "similarity_score": best_score
                        },
                    )

                    return response, best_score

            self._misses += 1
            self._update_stats("misses")
            return None

        except Exception as e:
            logger.error("semantic_cache_search_error", error=str(e))
            self._misses += 1
            return None

    def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """
        Get a semantically similar cached response.

        Args:
            request: LLM request

        Returns:
            Cached LLMResponse or None
        """
        result = self.find_similar(request)
        if result:
            return result[0]
        return None

    def set(
        self,
        request: LLMRequest,
        response: LLMResponse,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Add a response to the semantic cache.

        Args:
            request: Original LLM request
            response: LLM response to cache
            ttl: Optional TTL override

        Returns:
            True if cached successfully
        """
        prompt_text = self._get_prompt_text(request)
        embedding = self._generate_embedding(prompt_text)
        cache_id = self._get_cache_id()
        cache_ttl = ttl or self.ttl

        try:
            # Store response
            cache_key = f"{self.CACHE_PREFIX}{cache_id}"
            cache_data = {
                "content": response.content,
                "model": response.model,
                "provider": response.provider,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "created_at": response.created_at.isoformat(),
                "metadata": response.metadata,
                "original_prompt": prompt_text[:500],  # Store truncated prompt
            }

            self.redis.setex(
                cache_key,
                cache_ttl,
                json.dumps(cache_data)
            )

            # Store embedding
            emb_key = f"{self.EMBEDDING_PREFIX}{cache_id}"
            self.redis.setex(
                emb_key,
                cache_ttl,
                json.dumps(embedding)
            )

            # Add to index
            self.redis.sadd(self.INDEX_KEY, cache_id)

            # Set TTL on index entry (cleanup)
            self.redis.expire(self.INDEX_KEY, cache_ttl + 3600)

            logger.debug(
                "cache_set",
                cache_type="semantic",
                cache_id=cache_id,
                ttl=cache_ttl
            )

            return True

        except Exception as e:
            logger.error("semantic_cache_set_error", error=str(e))
            return False

    def clear(self) -> int:
        """Clear all semantic cache entries."""
        try:
            count = 0
            cached_ids = self.redis.smembers(self.INDEX_KEY)

            for cache_id in cached_ids:
                cache_key = f"{self.CACHE_PREFIX}{cache_id}"
                emb_key = f"{self.EMBEDDING_PREFIX}{cache_id}"
                count += self.redis.delete(cache_key, emb_key)

            self.redis.delete(self.INDEX_KEY)
            return count

        except Exception as e:
            logger.error("semantic_cache_clear_error", error=str(e))
            return 0

    def _update_stats(self, stat_type: str):
        """Update cache statistics in Redis."""
        try:
            self.redis.hincrby(self.STATS_KEY, stat_type, 1)
            self.redis.hincrby(self.STATS_KEY, "total", 1)
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats = self.redis.hgetall(self.STATS_KEY)
            hits = int(stats.get("hits", 0))
            misses = int(stats.get("misses", 0))
            total = hits + misses
            cache_size = self.redis.scard(self.INDEX_KEY)

            return {
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": hits / total if total > 0 else 0.0,
                "cache_size": cache_size,
                "similarity_threshold": self.similarity_threshold,
                "local_hits": self._hits,
                "local_misses": self._misses,
            }
        except Exception:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total": self._hits + self._misses,
                "hit_rate": self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0 else 0.0,
            }

    @property
    def hit_rate(self) -> float:
        """Calculate local hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
