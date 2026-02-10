"""
Cost tracking and analysis.
Tracks LLM costs and identifies optimization opportunities.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
import structlog

logger = structlog.get_logger()


@dataclass
class CostEntry:
    """A single cost entry."""
    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    cached: bool
    client_id: str
    trace_id: Optional[str] = None


@dataclass
class CostSummary:
    """Cost summary for a period."""
    total_cost: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    cost_by_provider: Dict[str, float]
    cost_by_model: Dict[str, float]
    cost_by_client: Dict[str, float]
    cache_savings: float
    cached_requests: int
    average_cost_per_request: float
    period_start: datetime
    period_end: datetime


class CostAnalyzer:
    """
    Analyzes LLM costs and identifies optimization opportunities.

    Features:
    - Track costs per request
    - Aggregate by provider, model, client
    - Calculate cache savings
    - Identify cost spikes
    - Generate cost reports
    """

    def __init__(self, retention_hours: int = 24):
        """
        Initialize cost analyzer.

        Args:
            retention_hours: Hours to retain cost data
        """
        self.retention_hours = retention_hours
        self._entries: List[CostEntry] = []
        self._lock = threading.Lock()

        # Baseline for spike detection
        self._hourly_baseline: Optional[float] = None
        self._daily_baseline: Optional[float] = None

    def record_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        cached: bool = False,
        client_id: str = "default",
        trace_id: Optional[str] = None
    ) -> None:
        """
        Record a cost entry.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
            cached: Whether response was cached
            client_id: Client identifier
            trace_id: Optional trace ID
        """
        entry = CostEntry(
            timestamp=datetime.utcnow(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            cached=cached,
            client_id=client_id,
            trace_id=trace_id,
        )

        with self._lock:
            self._entries.append(entry)
            self._cleanup_old_entries()

        logger.debug(
            "cost_recorded",
            provider=provider,
            model=model,
            cost=cost,
            cached=cached,
        )

    def _cleanup_old_entries(self) -> None:
        """Remove entries older than retention period."""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
        self._entries = [e for e in self._entries if e.timestamp > cutoff]

    def get_summary(
        self,
        hours: Optional[int] = None,
        provider: Optional[str] = None,
        client_id: Optional[str] = None
    ) -> CostSummary:
        """
        Get cost summary for a period.

        Args:
            hours: Number of hours to include (None for all)
            provider: Filter by provider
            client_id: Filter by client

        Returns:
            CostSummary with aggregated data
        """
        with self._lock:
            entries = self._entries.copy()

        # Filter by time
        if hours:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            entries = [e for e in entries if e.timestamp > cutoff]

        # Filter by provider
        if provider:
            entries = [e for e in entries if e.provider == provider]

        # Filter by client
        if client_id:
            entries = [e for e in entries if e.client_id == client_id]

        if not entries:
            now = datetime.utcnow()
            return CostSummary(
                total_cost=0.0,
                total_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                cost_by_provider={},
                cost_by_model={},
                cost_by_client={},
                cache_savings=0.0,
                cached_requests=0,
                average_cost_per_request=0.0,
                period_start=now,
                period_end=now,
            )

        # Aggregate
        total_cost = sum(e.cost for e in entries)
        total_requests = len(entries)
        total_input_tokens = sum(e.input_tokens for e in entries)
        total_output_tokens = sum(e.output_tokens for e in entries)

        cost_by_provider: Dict[str, float] = defaultdict(float)
        cost_by_model: Dict[str, float] = defaultdict(float)
        cost_by_client: Dict[str, float] = defaultdict(float)

        cached_requests = 0
        for entry in entries:
            cost_by_provider[entry.provider] += entry.cost
            cost_by_model[entry.model] += entry.cost
            cost_by_client[entry.client_id] += entry.cost
            if entry.cached:
                cached_requests += 1

        # Estimate cache savings (assume cached requests would cost same as average)
        avg_cost = total_cost / total_requests if total_requests > 0 else 0
        cache_savings = cached_requests * avg_cost

        return CostSummary(
            total_cost=round(total_cost, 6),
            total_requests=total_requests,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            cost_by_provider=dict(cost_by_provider),
            cost_by_model=dict(cost_by_model),
            cost_by_client=dict(cost_by_client),
            cache_savings=round(cache_savings, 6),
            cached_requests=cached_requests,
            average_cost_per_request=round(avg_cost, 8),
            period_start=min(e.timestamp for e in entries),
            period_end=max(e.timestamp for e in entries),
        )

    def check_cost_spike(self, threshold_multiplier: float = 1.5) -> Optional[Dict[str, Any]]:
        """
        Check for cost spikes compared to baseline.

        Args:
            threshold_multiplier: Alert if cost exceeds baseline * this

        Returns:
            Alert data if spike detected, None otherwise
        """
        current_hour = self.get_summary(hours=1)
        current_day = self.get_summary(hours=24)

        # Update baselines
        if self._hourly_baseline is None:
            self._hourly_baseline = current_hour.total_cost
            return None

        # Check hourly spike
        if current_hour.total_cost > self._hourly_baseline * threshold_multiplier:
            alert = {
                "type": "cost_spike",
                "severity": "warning",
                "current_hourly_cost": current_hour.total_cost,
                "baseline_hourly_cost": self._hourly_baseline,
                "multiplier": current_hour.total_cost / self._hourly_baseline
                if self._hourly_baseline > 0 else 0,
                "timestamp": datetime.utcnow().isoformat(),
            }
            logger.warning("cost_spike_detected", **alert)
            return alert

        # Update baseline (moving average)
        self._hourly_baseline = (
            self._hourly_baseline * 0.9 + current_hour.total_cost * 0.1
        )

        return None

    def get_top_spenders(self, limit: int = 10, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get top spending clients.

        Args:
            limit: Number of top spenders to return
            hours: Period to analyze

        Returns:
            List of top spenders with their costs
        """
        summary = self.get_summary(hours=hours)

        spenders = [
            {"client_id": client, "cost": cost}
            for client, cost in summary.cost_by_client.items()
        ]

        spenders.sort(key=lambda x: x["cost"], reverse=True)

        return spenders[:limit]

    def get_model_efficiency(self, hours: int = 24) -> Dict[str, Dict[str, float]]:
        """
        Analyze cost efficiency by model.

        Returns:
            Dict with cost per 1K tokens for each model
        """
        with self._lock:
            entries = self._entries.copy()

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        entries = [e for e in entries if e.timestamp > cutoff]

        model_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"cost": 0.0, "tokens": 0}
        )

        for entry in entries:
            model_stats[entry.model]["cost"] += entry.cost
            model_stats[entry.model]["tokens"] += entry.input_tokens + entry.output_tokens

        efficiency = {}
        for model, stats in model_stats.items():
            if stats["tokens"] > 0:
                efficiency[model] = {
                    "cost_per_1k_tokens": (stats["cost"] / stats["tokens"]) * 1000,
                    "total_cost": stats["cost"],
                    "total_tokens": stats["tokens"],
                }

        return efficiency

    def export_analysis(self, output_dir: str = "outputs") -> str:
        """Export cost analysis to JSON file."""
        analysis = {
            "summary_24h": self._summary_to_dict(self.get_summary(hours=24)),
            "summary_1h": self._summary_to_dict(self.get_summary(hours=1)),
            "top_spenders": self.get_top_spenders(),
            "model_efficiency": self.get_model_efficiency(),
            "exported_at": datetime.utcnow().isoformat(),
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = output_path / "cost_analysis.json"

        with open(filename, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info("cost_analysis_exported", filename=str(filename))

        return str(filename)

    def _summary_to_dict(self, summary: CostSummary) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "total_cost_usd": summary.total_cost,
            "total_requests": summary.total_requests,
            "total_input_tokens": summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "cost_by_provider": summary.cost_by_provider,
            "cost_by_model": summary.cost_by_model,
            "cache_savings_usd": summary.cache_savings,
            "cached_requests": summary.cached_requests,
            "average_cost_per_request": summary.average_cost_per_request,
            "period_start": summary.period_start.isoformat(),
            "period_end": summary.period_end.isoformat(),
        }


# Global cost analyzer
_analyzer: Optional[CostAnalyzer] = None


def get_cost_analyzer() -> CostAnalyzer:
    """Get the global cost analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = CostAnalyzer()
    return _analyzer
