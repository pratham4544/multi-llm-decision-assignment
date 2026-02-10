"""
Token counting utilities using tiktoken.
"""

from typing import List, Optional
import tiktoken

from src.providers.base import Message


class TokenCounter:
    """Token counter using tiktoken for accurate token counting."""

    # Cache encoders for reuse
    _encoders = {}

    @classmethod
    def get_encoder(cls, model: str = "gpt-3.5-turbo") -> tiktoken.Encoding:
        """
        Get or create a tiktoken encoder for the given model.

        Args:
            model: Model name (defaults to gpt-3.5-turbo for estimation)

        Returns:
            tiktoken.Encoding instance
        """
        if model not in cls._encoders:
            try:
                cls._encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base (used by GPT-4, GPT-3.5-turbo)
                cls._encoders[model] = tiktoken.get_encoding("cl100k_base")
        return cls._encoders[model]

    @classmethod
    def count_tokens(cls, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in a text string.

        Args:
            text: Text to count tokens in
            model: Model name for encoding selection

        Returns:
            Number of tokens
        """
        encoder = cls.get_encoder(model)
        return len(encoder.encode(text))

    @classmethod
    def count_message_tokens(
        cls,
        messages: List[Message],
        model: str = "gpt-3.5-turbo"
    ) -> int:
        """
        Count tokens in a list of chat messages.

        This includes overhead for message formatting (role, content separators).

        Args:
            messages: List of Message objects
            model: Model name for encoding selection

        Returns:
            Total token count
        """
        encoder = cls.get_encoder(model)
        tokens = 0

        # Each message has overhead for role, content markers
        # Approximate: <role> and <content> markers add ~4 tokens per message
        tokens_per_message = 4

        for message in messages:
            tokens += tokens_per_message
            tokens += len(encoder.encode(message.role))
            tokens += len(encoder.encode(message.content))

        # Additional overhead for the conversation
        tokens += 3  # priming tokens

        return tokens

    @classmethod
    def estimate_response_tokens(
        cls,
        prompt_tokens: int,
        avg_response_ratio: float = 1.5
    ) -> int:
        """
        Estimate expected response tokens based on prompt length.

        Args:
            prompt_tokens: Number of prompt tokens
            avg_response_ratio: Average ratio of response to prompt tokens

        Returns:
            Estimated response tokens
        """
        return int(prompt_tokens * avg_response_ratio)

    @classmethod
    def truncate_to_token_limit(
        cls,
        text: str,
        max_tokens: int,
        model: str = "gpt-3.5-turbo"
    ) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum token count
            model: Model name for encoding selection

        Returns:
            Truncated text
        """
        encoder = cls.get_encoder(model)
        tokens = encoder.encode(text)

        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return encoder.decode(truncated_tokens)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Convenience function for counting tokens."""
    return TokenCounter.count_tokens(text, model)


def count_message_tokens(messages: List[Message], model: str = "gpt-3.5-turbo") -> int:
    """Convenience function for counting message tokens."""
    return TokenCounter.count_message_tokens(messages, model)
