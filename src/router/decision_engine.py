"""
Routing decision logic and provider selection matrix.
Determines optimal provider/model based on complexity, health, and cost.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.providers.base import ComplexityLevel, BaseProvider
from src.providers.factory import ProviderFactory
from src.providers.health_monitor import get_health_monitor
from src.router.state import RouterContext


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    provider: str
    model: str
    fallback_chain: List[Tuple[str, str]]  # List of (provider, model) pairs
    reason: str
    estimated_cost: float
    estimated_latency_ms: float
    confidence: float  # 0-1 confidence in this decision


class DecisionEngine:
    """
    Engine for making routing decisions.

    Decision factors:
    1. Request complexity
    2. Provider health
    3. Cost optimization
    4. Latency requirements
    5. Model capabilities
    """

    # Model capabilities matrix
    # (provider, model) -> capabilities score for each complexity level
    MODEL_SCORES = {
        ("groq", "llama-3.1-8b-instant"): {
            ComplexityLevel.SIMPLE: 0.95,
            ComplexityLevel.MODERATE: 0.70,
            ComplexityLevel.COMPLEX: 0.50,
        },
        ("groq", "llama-3.3-70b-versatile"): {
            ComplexityLevel.SIMPLE: 0.90,
            ComplexityLevel.MODERATE: 0.95,
            ComplexityLevel.COMPLEX: 0.90,
        },
        ("groq", "moonshotai/kimi-k2-instruct-0905"): {
            ComplexityLevel.SIMPLE: 0.85,
            ComplexityLevel.MODERATE: 0.90,
            ComplexityLevel.COMPLEX: 0.80,
        },
        ("groq", "gemma2-9b-it"): {
            ComplexityLevel.SIMPLE: 0.90,
            ComplexityLevel.MODERATE: 0.75,
            ComplexityLevel.COMPLEX: 0.55,
        },
    }

    # Cost per 1K tokens (input + output average)
    MODEL_COSTS = {
        ("groq", "llama-3.1-8b-instant"): 0.000065,
        ("groq", "llama-3.3-70b-versatile"): 0.00069,
        ("groq", "moonshotai/kimi-k2-instruct-0905"): 0.00024,
        ("groq", "gemma2-9b-it"): 0.00020,
    }

    # Average latency (ms) per provider/model
    MODEL_LATENCY = {
        ("groq", "llama-3.1-8b-instant"): 200,
        ("groq", "llama-3.3-70b-versatile"): 500,
        ("groq", "moonshotai/kimi-k2-instruct-0905"): 400,
        ("groq", "gemma2-9b-it"): 250,
    }

    # Latency threshold (ms). Providers above this trigger gemma2 fallback.
    HIGH_LATENCY_THRESHOLD_MS = 1000

    def __init__(self):
        self.health_monitor = get_health_monitor()

    def decide(
        self,
        complexity: ComplexityLevel,
        context: Optional[RouterContext] = None,
        token_count: int = 0
    ) -> RoutingDecision:
        """
        Make a routing decision based on complexity and context.

        Uses priority-based deterministic rules first, then falls back
        to score-based selection for unmatched cases.

        Routing rules (checked in order):
        1. High latency detected -> gemma2-9b-it (fast fallback)
        2. High API request rate -> gemma2-9b-it (conserve quota)
        3. Coding tasks -> mixtral-8x7b-32768 (code specialist)
        4. Complex/analytical -> llama-3.3-70b-versatile (most capable)
        5. Simple queries -> llama-3.1-8b-instant (fast & cheap)
        6. Fallback -> score-based selection
        """
        context = context or RouterContext()

        # Get available providers
        healthy_providers = self._get_healthy_providers()

        if not healthy_providers:
            return RoutingDecision(
                provider="groq",
                model="llama-3.1-8b-instant",
                fallback_chain=[],
                reason="No healthy providers, using emergency fallback",
                estimated_cost=0.0,
                estimated_latency_ms=0.0,
                confidence=0.3,
            )

        question_type = context.question_type or "simple"

        # --- DETERMINISTIC ROUTING RULES (priority order) ---

        # Rule 1: High latency fallback -> gemma2-9b-it
        high_latency_model = self._check_high_latency()
        if high_latency_model:
            if self._is_model_available("groq", "gemma2-9b-it", healthy_providers):
                return self._build_decision(
                    "groq", "gemma2-9b-it", complexity, token_count,
                    reason=f"Latency fallback: {high_latency_model} showing high latency, routing to gemma2-9b-it",
                    confidence=0.85,
                )

        # Rule 2: High request rate -> gemma2-9b-it
        if context.high_request_rate:
            if self._is_model_available("groq", "gemma2-9b-it", healthy_providers):
                return self._build_decision(
                    "groq", "gemma2-9b-it", complexity, token_count,
                    reason="High request rate detected, routing to gemma2-9b-it to conserve quota",
                    confidence=0.80,
                )

        # Rule 3: Coding tasks -> moonshotai/kimi-k2-instruct-0905 (code specialist)
        if question_type == "coding":
            if self._is_model_available("groq", "moonshotai/kimi-k2-instruct-0905", healthy_providers):
                return self._build_decision(
                    "groq", "moonshotai/kimi-k2-instruct-0905", complexity, token_count,
                    reason="Coding task routed to moonshotai/kimi-k2-instruct-0905 (code specialist)",
                    confidence=0.90,
                )

        # Rule 4: Complex/analytical -> llama-3.3-70b-versatile
        if complexity == ComplexityLevel.COMPLEX or question_type == "analytical":
            if self._is_model_available("groq", "llama-3.3-70b-versatile", healthy_providers):
                return self._build_decision(
                    "groq", "llama-3.3-70b-versatile", complexity, token_count,
                    reason="Complex/analytical request routed to llama-3.3-70b-versatile",
                    confidence=0.90,
                )

        # Rule 5: Simple queries -> llama-3.1-8b-instant
        if complexity == ComplexityLevel.SIMPLE and question_type == "simple":
            if self._is_model_available("groq", "llama-3.1-8b-instant", healthy_providers):
                return self._build_decision(
                    "groq", "llama-3.1-8b-instant", complexity, token_count,
                    reason="Simple request routed to fast llama-3.1-8b-instant",
                    confidence=0.95,
                )

        # --- FALLBACK: Score-based selection for remaining cases ---
        scored_options = []
        for provider in healthy_providers:
            models = self._get_provider_models(provider)
            for model in models:
                score = self._score_option(
                    provider=provider,
                    model=model,
                    complexity=complexity,
                    context=context,
                    token_count=token_count
                )
                scored_options.append((provider, model, score))

        scored_options.sort(key=lambda x: x[2], reverse=True)

        if not scored_options:
            return RoutingDecision(
                provider="groq",
                model="llama-3.1-8b-instant",
                fallback_chain=[],
                reason="No scored options, using default",
                estimated_cost=0.0,
                estimated_latency_ms=0.0,
                confidence=0.5,
            )

        best_provider, best_model, best_score = scored_options[0]
        fallback_chain = [(p, m) for p, m, s in scored_options[1:4]]

        key = (best_provider, best_model)
        estimated_cost = self._estimate_cost(key, token_count)
        estimated_latency = self.MODEL_LATENCY.get(key, 500)

        return RoutingDecision(
            provider=best_provider,
            model=best_model,
            fallback_chain=fallback_chain,
            reason=self._generate_reason(complexity, best_provider, best_model),
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            confidence=min(best_score, 1.0),
        )

    def _get_healthy_providers(self) -> List[str]:
        """Get list of healthy provider names."""
        return self.health_monitor.get_healthy_providers()

    def _get_provider_models(self, provider: str) -> List[str]:
        """Get available models for a provider."""
        provider_instance = ProviderFactory.get_provider(provider)
        if provider_instance:
            return [m.model_id for m in provider_instance.get_available_models()]
        return []

    def _check_high_latency(self) -> Optional[str]:
        """Check if any provider is showing high latency.

        Returns the name of the high-latency provider, or None.
        """
        all_health = self.health_monitor.get_all_health()
        for provider_name, health in all_health.items():
            if (health.latency_ms is not None
                    and health.latency_ms > self.HIGH_LATENCY_THRESHOLD_MS):
                return provider_name
        return None

    def _is_model_available(
        self, provider: str, model: str, healthy_providers: List[str]
    ) -> bool:
        """Check if a specific provider/model combination is available."""
        if provider not in healthy_providers:
            return False
        models = self._get_provider_models(provider)
        return model in models

    def _build_decision(
        self,
        provider: str,
        model: str,
        complexity: ComplexityLevel,
        token_count: int,
        reason: str,
        confidence: float,
    ) -> RoutingDecision:
        """Build a RoutingDecision with an auto-generated fallback chain."""
        key = (provider, model)
        estimated_cost = self._estimate_cost(key, token_count)
        estimated_latency = self.MODEL_LATENCY.get(key, 500)

        # Fallback chain: all other models except the selected one
        fallback_chain = [
            (p, m) for (p, m) in self.MODEL_SCORES.keys()
            if not (p == provider and m == model)
        ][:3]

        return RoutingDecision(
            provider=provider,
            model=model,
            fallback_chain=fallback_chain,
            reason=reason,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            confidence=confidence,
        )

    def _score_option(
        self,
        provider: str,
        model: str,
        complexity: ComplexityLevel,
        context: RouterContext,
        token_count: int
    ) -> float:
        """
        Score a provider/model option.

        Score components:
        - Quality score (40%): How well the model handles this complexity
        - Cost score (30%): Lower cost = higher score
        - Latency score (20%): Lower latency = higher score
        - Health score (10%): Provider health status
        """
        key = (provider, model)

        # Quality score (0-1)
        quality_scores = self.MODEL_SCORES.get(key, {})
        quality = quality_scores.get(complexity, 0.5)

        # Cost score (0.3-1.0, inverted - lower cost = higher score)
        # Uses min-max normalization with a floor so no model gets zero
        cost = self.MODEL_COSTS.get(key, 0.001)
        min_cost = min(self.MODEL_COSTS.values())
        max_cost = max(self.MODEL_COSTS.values())
        if max_cost > min_cost:
            cost_score = 1.0 - 0.7 * ((cost - min_cost) / (max_cost - min_cost))
        else:
            cost_score = 0.5

        # Check cost constraint
        if context.max_cost:
            estimated = self._estimate_cost(key, token_count)
            if estimated > context.max_cost:
                cost_score *= 0.3  # Heavily penalize

        # Latency score (0.3-1.0, inverted - lower latency = higher score)
        latency = self.MODEL_LATENCY.get(key, 500)
        min_latency = min(self.MODEL_LATENCY.values())
        max_latency = max(self.MODEL_LATENCY.values())
        if max_latency > min_latency:
            latency_score = 1.0 - 0.7 * ((latency - min_latency) / (max_latency - min_latency))
        else:
            latency_score = 0.5

        # Check latency constraint
        if context.max_latency_ms and latency > context.max_latency_ms:
            latency_score *= 0.3  # Heavily penalize

        # Health score
        health = self.health_monitor.get_health(provider)
        if health:
            health_score = health.success_rate
        else:
            health_score = 1.0  # Assume healthy if not checked

        # Weight adjustments based on complexity and context
        # Complex requests prioritize quality; simple requests prioritize cost/speed
        complexity_boost = {
            ComplexityLevel.SIMPLE: 0.0,
            ComplexityLevel.MODERATE: 0.3,
            ComplexityLevel.COMPLEX: 0.6,
        }.get(complexity, 0.0)

        quality_weight = (0.4 + complexity_boost) * (1 - context.cost_priority)
        cost_weight = (0.3 - complexity_boost * 0.3) * (1 + context.cost_priority)
        latency_weight = (0.2 - complexity_boost * 0.2) * (1 + context.latency_priority)
        health_weight = 0.1

        # Normalize weights
        total_weight = quality_weight + cost_weight + latency_weight + health_weight
        quality_weight /= total_weight
        cost_weight /= total_weight
        latency_weight /= total_weight
        health_weight /= total_weight

        # Calculate final score
        score = (
            quality * quality_weight +
            cost_score * cost_weight +
            latency_score * latency_weight +
            health_score * health_weight
        )

        return score

    def _estimate_cost(self, key: Tuple[str, str], token_count: int) -> float:
        """Estimate cost for a request."""
        cost_per_1k = self.MODEL_COSTS.get(key, 0.001)
        # Assume output tokens = 1.5x input
        total_tokens = token_count * 2.5
        return (total_tokens / 1000) * cost_per_1k

    def _generate_reason(
        self,
        complexity: ComplexityLevel,
        provider: str,
        model: str
    ) -> str:
        """Generate human-readable reason for the decision."""
        reasons = {
            ComplexityLevel.SIMPLE: f"Simple request routed to fast, cost-efficient {model}",
            ComplexityLevel.MODERATE: f"Moderate complexity handled by balanced {model}",
            ComplexityLevel.COMPLEX: f"Complex request requires capable {model}",
        }
        return reasons.get(complexity, f"Routed to {provider}/{model}")

    def get_fallback(
        self,
        current_provider: str,
        current_model: str,
        complexity: ComplexityLevel,
        exclude: Optional[List[str]] = None
    ) -> Optional[Tuple[str, str]]:
        """
        Get next fallback option after a failure.

        Args:
            current_provider: Provider that failed
            current_model: Model that failed
            complexity: Request complexity
            exclude: List of providers to exclude

        Returns:
            Tuple of (provider, model) or None if no fallback available
        """
        exclude = exclude or []
        exclude.append(current_provider)

        # Get remaining healthy providers
        healthy = [p for p in self._get_healthy_providers() if p not in exclude]

        if not healthy:
            # Try same provider with different model
            models = self._get_provider_models(current_provider)
            for model in models:
                if model != current_model:
                    return (current_provider, model)
            return None

        # Get best option from remaining providers
        decision = self.decide(
            complexity=complexity,
            context=RouterContext(healthy_providers=healthy)
        )

        if decision.provider not in exclude:
            return (decision.provider, decision.model)

        return None


# Global decision engine instance
_decision_engine: Optional[DecisionEngine] = None


def get_decision_engine() -> DecisionEngine:
    """Get the global decision engine instance."""
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = DecisionEngine()
    return _decision_engine
