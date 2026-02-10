"""
Provider factory for creating and managing provider instances.
"""

from typing import Dict, List, Optional, Type

from src.config import settings
from src.providers.base import BaseProvider, ComplexityLevel


class ProviderFactory:
    """Factory for creating and managing LLM provider instances."""

    _providers: Dict[str, BaseProvider] = {}
    _provider_classes: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseProvider]):
        """Register a provider class."""
        cls._provider_classes[name] = provider_class

    @classmethod
    def get_provider(cls, name: str) -> Optional[BaseProvider]:
        """Get a provider instance by name."""
        if name not in cls._providers:
            cls._providers[name] = cls._create_provider(name)
        return cls._providers.get(name)

    @classmethod
    def _create_provider(cls, name: str) -> Optional[BaseProvider]:
        """Create a new provider instance."""
        if name == "groq":
            from src.providers.groq_provider import GroqProvider
            if settings.groq_api_key:
                return GroqProvider(api_key=settings.groq_api_key)

        # Add other providers here as they're implemented
        # elif name == "openai":
        #     from src.providers.openai_provider import OpenAIProvider
        #     if settings.openai_api_key:
        #         return OpenAIProvider(api_key=settings.openai_api_key)

        # elif name == "anthropic":
        #     from src.providers.anthropic_provider import AnthropicProvider
        #     if settings.anthropic_api_key:
        #         return AnthropicProvider(api_key=settings.anthropic_api_key)

        return None

    @classmethod
    def get_all_providers(cls) -> Dict[str, BaseProvider]:
        """Get all available provider instances."""
        # Initialize all configured providers
        available_providers = ["groq"]  # Add more as implemented

        for name in available_providers:
            if name not in cls._providers:
                provider = cls._create_provider(name)
                if provider:
                    cls._providers[name] = provider

        return cls._providers

    @classmethod
    def get_available_provider_names(cls) -> List[str]:
        """Get list of available provider names."""
        providers = cls.get_all_providers()
        return list(providers.keys())

    @classmethod
    def get_provider_for_complexity(
        cls,
        complexity: ComplexityLevel,
        exclude: Optional[List[str]] = None
    ) -> Optional[BaseProvider]:
        """
        Get a suitable provider for the given complexity level.

        Args:
            complexity: Request complexity level
            exclude: List of provider names to exclude

        Returns:
            A suitable provider or None
        """
        exclude = exclude or []
        providers = cls.get_all_providers()

        # Priority order: groq (cheapest for simple), then others
        priority_order = ["groq", "openai", "anthropic"]

        for name in priority_order:
            if name in providers and name not in exclude:
                return providers[name]

        return None

    @classmethod
    def get_fallback_chain(
        cls,
        primary: str,
        complexity: ComplexityLevel
    ) -> List[BaseProvider]:
        """
        Get a fallback chain of providers.

        Args:
            primary: Primary provider name
            complexity: Request complexity level

        Returns:
            List of providers in fallback order
        """
        providers = cls.get_all_providers()
        chain = []

        # Start with primary
        if primary in providers:
            chain.append(providers[primary])

        # Add fallbacks based on complexity
        fallback_order = ["groq", "openai", "anthropic"]

        for name in fallback_order:
            if name != primary and name in providers:
                chain.append(providers[name])

        return chain

    @classmethod
    def reset(cls):
        """Reset all provider instances (mainly for testing)."""
        cls._providers.clear()


# Convenience functions
def get_provider(name: str) -> Optional[BaseProvider]:
    """Get a provider by name."""
    return ProviderFactory.get_provider(name)


def get_all_providers() -> Dict[str, BaseProvider]:
    """Get all available providers."""
    return ProviderFactory.get_all_providers()


def get_default_provider() -> Optional[BaseProvider]:
    """Get the default provider (currently Groq)."""
    return ProviderFactory.get_provider("groq")
