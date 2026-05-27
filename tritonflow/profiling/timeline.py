"""Execution timeline capture with Chrome trace export."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

__all__ = ["TimelineEvent", "ExecutionTimeline"]


@dataclass
class TimelineEvent:
    """A single recorded timeline event."""

    name: str
    start_ms: float
    end_ms: float
    category: str = "kernel"
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class ExecutionTimeline:
    """Record and export kernel execution timelines."""

    def __init__(self) -> None:
        self.events: list[TimelineEvent] = []
        self._start_time: float | None = None

    def start(self) -> None:
        """Set the timeline reference start time."""
        self._start_time = time.perf_counter()

    def _elapsed_ms(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Call start() before recording events.")
        return (time.perf_counter() - self._start_time) * 1000.0

    @contextmanager
    def record(self, name: str, category: str = "kernel"):
        """Record a timeline event."""
        start_ms = self._elapsed_ms()
        yield
        end_ms = self._elapsed_ms()
        self.events.append(
            TimelineEvent(name=name, start_ms=start_ms, end_ms=end_ms, category=category)
        )

    def to_chrome_trace(self) -> dict:
        """Export to Chrome trace format (chrome://tracing).

        Returns a dict compatible with the Chrome Trace Event Format.
        """
        trace_events = []
        for evt in self.events:
            trace_events.append(
                {
                    "name": evt.name,
                    "cat": evt.category,
                    "ph": "X",  # complete event
                    "ts": evt.start_ms * 1000.0,  # microseconds
                    "dur": evt.duration_ms * 1000.0,
                    "pid": 1,
                    "tid": 1,
                    "args": evt.metadata,
                }
            )
        return {"traceEvents": trace_events}

    def summary(self) -> str:
        """Formatted timeline summary."""
        if not self.events:
            return "No events recorded."

        header = f"{'Event':<30} {'Category':<12} {'Start (ms)':>12} {'Duration (ms)':>14}"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for evt in self.events:
            lines.append(
                f"{evt.name:<30} {evt.category:<12} {evt.start_ms:>12.3f} {evt.duration_ms:>14.3f}"
            )
        lines.append(sep)

        total = sum(e.duration_ms for e in self.events)
        lines.append(f"Total: {total:.3f} ms across {len(self.events)} events")
        return "\n".join(lines)
