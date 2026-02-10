"""
Distributed tracing implementation.
Tracks request lifecycle across components.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import structlog

from src.config import settings

logger = structlog.get_logger()


@dataclass
class SpanEvent:
    """An event within a span."""
    name: str
    timestamp: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A span representing a unit of work."""
    name: str
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    start_time: str
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return asdict(self)


class Tracer:
    """
    Simple distributed tracer for request tracking.

    Features:
    - Generate unique trace IDs
    - Track spans with parent-child relationships
    - Record events and attributes
    - Export traces to JSON
    """

    def __init__(self, service_name: str = "llm-router"):
        self.service_name = service_name
        self._traces: Dict[str, List[Span]] = {}
        self._active_spans: Dict[str, Span] = {}

    def generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return str(uuid.uuid4())

    def generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return uuid.uuid4().hex[:16]

    def start_span(
        self,
        name: str,
        trace_id: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """
        Start a new span.

        Args:
            name: Span name
            trace_id: Trace ID this span belongs to
            parent_id: Optional parent span ID
            attributes: Initial attributes

        Returns:
            New Span instance
        """
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=self.generate_span_id(),
            parent_id=parent_id,
            start_time=datetime.utcnow().isoformat(),
            attributes=attributes or {},
        )

        # Store span
        if trace_id not in self._traces:
            self._traces[trace_id] = []
        self._traces[trace_id].append(span)
        self._active_spans[span.span_id] = span

        logger.debug(
            "span_started",
            trace_id=trace_id,
            span_id=span.span_id,
            span_name=name,
        )

        return span

    def end_span(
        self,
        span: Span,
        status: str = "ok",
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End a span.

        Args:
            span: Span to end
            status: Final status (ok, error)
            attributes: Additional attributes to add
        """
        end_time = datetime.utcnow()
        span.end_time = end_time.isoformat()
        span.status = status

        # Calculate duration
        start = datetime.fromisoformat(span.start_time)
        span.duration_ms = (end_time - start).total_seconds() * 1000

        if attributes:
            span.attributes.update(attributes)

        # Remove from active spans
        self._active_spans.pop(span.span_id, None)

        logger.debug(
            "span_ended",
            trace_id=span.trace_id,
            span_id=span.span_id,
            span_name=span.name,
            duration_ms=round(span.duration_ms, 2),
            status=status,
        )

    def add_event(
        self,
        span: Span,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an event to a span.

        Args:
            span: Span to add event to
            name: Event name
            attributes: Event attributes
        """
        event = SpanEvent(
            name=name,
            timestamp=datetime.utcnow().isoformat(),
            attributes=attributes or {},
        )
        span.events.append(event)

    def set_attribute(self, span: Span, key: str, value: Any) -> None:
        """Set an attribute on a span."""
        span.attributes[key] = value

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return self._traces.get(trace_id, [])

    def export_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Export a trace as a dictionary.

        Args:
            trace_id: Trace ID to export

        Returns:
            Trace data as dictionary
        """
        spans = self.get_trace(trace_id)

        return {
            "trace_id": trace_id,
            "service": self.service_name,
            "spans": [span.to_dict() for span in spans],
            "span_count": len(spans),
            "exported_at": datetime.utcnow().isoformat(),
        }

    def save_trace(self, trace_id: str, output_dir: str = "outputs") -> str:
        """
        Save a trace to a JSON file.

        Args:
            trace_id: Trace ID to save
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        trace_data = self.export_trace(trace_id)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = output_path / f"trace_{trace_id[:8]}.json"

        with open(filename, "w") as f:
            json.dump(trace_data, f, indent=2)

        logger.info("trace_saved", trace_id=trace_id, filename=str(filename))

        return str(filename)

    def clear_trace(self, trace_id: str) -> None:
        """Clear a trace from memory."""
        self._traces.pop(trace_id, None)


class SpanContext:
    """Context manager for spans."""

    def __init__(
        self,
        tracer: Tracer,
        name: str,
        trace_id: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.tracer = tracer
        self.name = name
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.attributes = attributes or {}
        self.span: Optional[Span] = None

    def __enter__(self) -> Span:
        self.span = self.tracer.start_span(
            name=self.name,
            trace_id=self.trace_id,
            parent_id=self.parent_id,
            attributes=self.attributes,
        )
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "error" if exc_type else "ok"
        attributes = {}
        if exc_val:
            attributes["error"] = str(exc_val)
            attributes["error_type"] = exc_type.__name__ if exc_type else None

        self.tracer.end_span(self.span, status=status, attributes=attributes)
        return False


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def trace_span(
    name: str,
    trace_id: str,
    parent_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> SpanContext:
    """
    Create a span context manager.

    Usage:
        with trace_span("operation", trace_id) as span:
            # do work
            span.attributes["result"] = "success"
    """
    return SpanContext(
        tracer=get_tracer(),
        name=name,
        trace_id=trace_id,
        parent_id=parent_id,
        attributes=attributes,
    )
