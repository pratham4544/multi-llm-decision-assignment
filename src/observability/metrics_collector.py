"""
Metrics collection and aggregation.
Real-time metrics tracking for the LLM router.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
import structlog

logger = structlog.get_logger()


@dataclass
class MetricPoint:
    """A single metric data point."""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """A monotonically increasing counter."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter."""
        with self._lock:
            self._value += amount

    def get(self) -> float:
        """Get current value."""
        return self._value

    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


class Gauge:
    """A metric that can go up and down."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set the gauge value."""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1) -> None:
        """Increment the gauge."""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1) -> None:
        """Decrement the gauge."""
        with self._lock:
            self._value -= amount

    def get(self) -> float:
        """Get current value."""
        return self._value


class Histogram:
    """A histogram for tracking value distributions."""

    DEFAULT_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None
    ):
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._values: List[float] = []
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Record a value."""
        with self._lock:
            self._values.append(value)
            # Keep only last 10000 values for memory efficiency
            if len(self._values) > 10000:
                self._values = self._values[-10000:]

    def get_percentile(self, percentile: float) -> float:
        """Get a percentile value (0-100)."""
        if not self._values:
            return 0.0
        sorted_values = sorted(self._values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_stats(self) -> Dict[str, float]:
        """Get histogram statistics."""
        if not self._values:
            return {"count": 0, "sum": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

        return {
            "count": len(self._values),
            "sum": sum(self._values),
            "avg": sum(self._values) / len(self._values),
            "min": min(self._values),
            "max": max(self._values),
            "p50": self.get_percentile(50),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
        }

    def reset(self) -> None:
        """Reset histogram."""
        with self._lock:
            self._values.clear()


class MetricsCollector:
    """
    Centralized metrics collection for the LLM router.

    Tracks:
    - Request counts by provider, model, client
    - Latency distributions
    - Cache hit/miss rates
    - Error rates
    - Cost metrics
    """

    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()

        # Initialize standard metrics
        self._init_standard_metrics()

    def _init_standard_metrics(self):
        """Initialize standard metrics."""
        # Request counters
        self.create_counter("requests_total", "Total number of requests")
        self.create_counter("requests_success", "Successful requests")
        self.create_counter("requests_error", "Failed requests")
        self.create_counter("requests_cached", "Requests served from cache")

        # Provider counters
        self.create_counter("provider_requests", "Requests by provider")
        self.create_counter("provider_errors", "Errors by provider")

        # Cache counters
        self.create_counter("cache_hits_exact", "Exact cache hits")
        self.create_counter("cache_hits_semantic", "Semantic cache hits")
        self.create_counter("cache_misses", "Cache misses")

        # Latency histograms
        self.create_histogram(
            "request_latency_ms",
            "Request latency in milliseconds",
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        )
        self.create_histogram(
            "provider_latency_ms",
            "Provider latency in milliseconds",
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000]
        )

        # Cost gauge
        self.create_gauge("total_cost_usd", "Total cost in USD")
        self.create_gauge("active_requests", "Currently active requests")

        # Token counters
        self.create_counter("tokens_input", "Total input tokens")
        self.create_counter("tokens_output", "Total output tokens")

    def create_counter(self, name: str, description: str = "") -> Counter:
        """Create or get a counter."""
        if name not in self._counters:
            self._counters[name] = Counter(name, description)
        return self._counters[name]

    def create_gauge(self, name: str, description: str = "") -> Gauge:
        """Create or get a gauge."""
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description)
        return self._gauges[name]

    def create_histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """Create or get a histogram."""
        if name not in self._histograms:
            self._histograms[name] = Histogram(name, description, buckets)
        return self._histograms[name]

    def inc_counter(self, name: str, amount: float = 1) -> None:
        """Increment a counter."""
        if name in self._counters:
            self._counters[name].inc(amount)

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        if name in self._gauges:
            self._gauges[name].set(value)

    def observe_histogram(self, name: str, value: float) -> None:
        """Record a histogram value."""
        if name in self._histograms:
            self._histograms[name].observe(value)

    def record_request(
        self,
        provider: str,
        model: str,
        latency_ms: float,
        success: bool,
        cached: bool = False,
        cache_type: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0
    ) -> None:
        """
        Record metrics for a completed request.

        Args:
            provider: Provider name
            model: Model name
            latency_ms: Request latency
            success: Whether request succeeded
            cached: Whether response was cached
            cache_type: Type of cache hit
            input_tokens: Input token count
            output_tokens: Output token count
            cost: Request cost
        """
        self.inc_counter("requests_total")

        if success:
            self.inc_counter("requests_success")
        else:
            self.inc_counter("requests_error")
            self.inc_counter("provider_errors")

        if cached:
            self.inc_counter("requests_cached")
            if cache_type == "exact":
                self.inc_counter("cache_hits_exact")
            elif cache_type == "semantic":
                self.inc_counter("cache_hits_semantic")
        else:
            self.inc_counter("cache_misses")

        self.inc_counter("provider_requests")
        self.inc_counter("tokens_input", input_tokens)
        self.inc_counter("tokens_output", output_tokens)

        self.observe_histogram("request_latency_ms", latency_ms)
        self.observe_histogram("provider_latency_ms", latency_ms)

        # Update cost gauge
        current_cost = self._gauges.get("total_cost_usd")
        if current_cost:
            current_cost.inc(cost)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        metrics = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        for name, counter in self._counters.items():
            metrics["counters"][name] = counter.get()

        for name, gauge in self._gauges.items():
            metrics["gauges"][name] = gauge.get()

        for name, histogram in self._histograms.items():
            metrics["histograms"][name] = histogram.get_stats()

        return metrics

    def export_metrics(self, output_dir: str = "outputs") -> str:
        """Export metrics to JSON file."""
        metrics = self.get_metrics()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = output_path / "metrics_snapshot.json"

        with open(filename, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("metrics_exported", filename=str(filename))

        return str(filename)

    def reset_all(self) -> None:
        """Reset all metrics."""
        for counter in self._counters.values():
            counter.reset()
        for gauge in self._gauges.values():
            gauge.set(0)
        for histogram in self._histograms.values():
            histogram.reset()


# Global metrics collector
_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
