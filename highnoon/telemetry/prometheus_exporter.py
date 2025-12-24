"""
Prometheus metrics exporter for HSMN architecture observability.

This module provides enterprise-grade telemetry collection for TensorFlow custom ops,
controllers, and training dynamics. Metrics are exposed via Prometheus format for
integration with Grafana dashboards and alerting systems.

Phase 9.2 Deliverable: Comprehensive monitoring infrastructure per technical_roadmap.md.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _env_flag(name: str, default: str = "1") -> bool:
    """Parse environment variable as boolean flag."""
    return os.getenv(name, default).strip().lower() not in {"0", "false", "off", "no"}


@dataclass
class MetricConfig:
    """Configuration for metrics collection."""

    enabled: bool = _env_flag("HSMN_METRICS_ENABLED", "1")
    export_interval_sec: float = float(os.getenv("HSMN_METRICS_INTERVAL", "10.0"))
    export_path: Path = Path(os.getenv("HSMN_METRICS_PATH", "state/metrics.prom"))
    histogram_buckets: tuple[float, ...] = (
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
        1.0,
        5.0,
        10.0,
    )
    enable_debug_logging: bool = _env_flag("HSMN_METRICS_DEBUG", "0")


@dataclass
class Counter:
    """Prometheus counter metric (monotonically increasing)."""

    name: str
    help_text: str
    labels: dict[str, str] = field(default_factory=dict)
    value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        """Increment counter by amount."""
        self.value += amount

    def to_prometheus(self) -> str:
        """Format as Prometheus exposition format."""
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
        label_part = f"{{{label_str}}}" if label_str else ""
        return f"{self.name}{label_part} {self.value}"


@dataclass
class Gauge:
    """Prometheus gauge metric (can increase or decrease)."""

    name: str
    help_text: str
    labels: dict[str, str] = field(default_factory=dict)
    value: float = 0.0

    def set(self, value: float) -> None:
        """Set gauge to specific value."""
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment gauge."""
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge."""
        self.value -= amount

    def to_prometheus(self) -> str:
        """Format as Prometheus exposition format."""
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
        label_part = f"{{{label_str}}}" if label_str else ""
        return f"{self.name}{label_part} {self.value}"


@dataclass
class Histogram:
    """Prometheus histogram metric (distribution of observations)."""

    name: str
    help_text: str
    labels: dict[str, str] = field(default_factory=dict)
    buckets: tuple[float, ...] = field(default_factory=tuple)
    bucket_counts: dict[float, int] = field(default_factory=dict)
    sum_value: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        """Initialize bucket counts."""
        if not self.bucket_counts:
            self.bucket_counts = dict.fromkeys(self.buckets, 0)
            self.bucket_counts[float("inf")] = 0

    def observe(self, value: float) -> None:
        """Add observation to histogram."""
        self.sum_value += value
        self.count += 1
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1
        self.bucket_counts[float("inf")] += 1

    def to_prometheus(self) -> str:
        """Format as Prometheus exposition format."""
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
        label_part = f"{{{label_str}}}" if label_str else ""
        lines = []
        for bucket, count in sorted(self.bucket_counts.items()):
            bucket_str = "+Inf" if bucket == float("inf") else str(bucket)
            bucket_label = f',le="{bucket_str}"' if label_str else f'le="{bucket_str}"'
            if label_str:
                lines.append(f"{self.name}_bucket{{{label_str},{bucket_label[1:]}}} {count}")
            else:
                lines.append(f"{self.name}_bucket{{{bucket_label[1:]}}} {count}")
        lines.append(f"{self.name}_sum{label_part} {self.sum_value}")
        lines.append(f"{self.name}_count{label_part} {self.count}")
        return "\n".join(lines)


class MetricsRegistry:
    """
    Thread-safe registry for Prometheus metrics.

    Maintains counters, gauges, and histograms for all HSMN components.
    Automatically exports metrics to disk at configurable intervals.
    """

    def __init__(self, config: MetricConfig | None = None) -> None:
        self.config = config or MetricConfig()
        self._lock = threading.RLock()
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._last_export = 0.0
        self._export_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def counter(self, name: str, help_text: str, labels: dict[str, str] | None = None) -> Counter:
        """Get or create a counter metric."""
        labels = labels or {}
        key = self._metric_key(name, labels)
        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter(name, help_text, labels)
            return self._counters[key]

    def gauge(self, name: str, help_text: str, labels: dict[str, str] | None = None) -> Gauge:
        """Get or create a gauge metric."""
        labels = labels or {}
        key = self._metric_key(name, labels)
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = Gauge(name, help_text, labels)
            return self._gauges[key]

    def histogram(
        self,
        name: str,
        help_text: str,
        labels: dict[str, str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        """Get or create a histogram metric."""
        labels = labels or {}
        buckets = buckets or self.config.histogram_buckets
        key = self._metric_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                hist = Histogram(name, help_text, labels, buckets)
                hist.__post_init__()
                self._histograms[key] = hist
            return self._histograms[key]

    def _metric_key(self, name: str, labels: dict[str, str]) -> str:
        """Generate unique key for metric with labels."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}" if label_str else name

    def export_to_file(self) -> None:
        """Export all metrics to Prometheus format file."""
        if not self.config.enabled:
            return

        lines = [
            "# HSMN Architecture Metrics Export",
            f"# Generated at {datetime.now(timezone.utc).isoformat()}",
            "",
        ]

        with self._lock:
            # Export counters
            counter_names = {c.name for c in self._counters.values()}
            for counter_name in sorted(counter_names):
                counters = [c for c in self._counters.values() if c.name == counter_name]
                if counters:
                    lines.append(f"# HELP {counter_name} {counters[0].help_text}")
                    lines.append(f"# TYPE {counter_name} counter")
                    for counter in counters:
                        lines.append(counter.to_prometheus())
                    lines.append("")

            # Export gauges
            gauge_names = {g.name for g in self._gauges.values()}
            for gauge_name in sorted(gauge_names):
                gauges = [g for g in self._gauges.values() if g.name == gauge_name]
                if gauges:
                    lines.append(f"# HELP {gauge_name} {gauges[0].help_text}")
                    lines.append(f"# TYPE {gauge_name} gauge")
                    for gauge in gauges:
                        lines.append(gauge.to_prometheus())
                    lines.append("")

            # Export histograms
            hist_names = {h.name for h in self._histograms.values()}
            for hist_name in sorted(hist_names):
                hists = [h for h in self._histograms.values() if h.name == hist_name]
                if hists:
                    lines.append(f"# HELP {hist_name} {hists[0].help_text}")
                    lines.append(f"# TYPE {hist_name} histogram")
                    for hist in hists:
                        lines.append(hist.to_prometheus())
                    lines.append("")

        # Atomic write
        path = self.config.export_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text("\n".join(lines))
        tmp_path.replace(path)

        if self.config.enable_debug_logging:
            print(f"[MetricsRegistry] Exported {len(lines)} lines to {path}")

    def start_auto_export(self) -> None:
        """Start background thread for periodic metric export."""
        if not self.config.enabled:
            return
        if self._export_thread and self._export_thread.is_alive():
            return

        self._stop_event.clear()
        self._export_thread = threading.Thread(
            target=self._export_loop, name="MetricsExporter", daemon=True
        )
        self._export_thread.start()

    def stop_auto_export(self) -> None:
        """Stop background export thread."""
        self._stop_event.set()
        if self._export_thread and self._export_thread.is_alive():
            self._export_thread.join(timeout=2.0)

    def _export_loop(self) -> None:
        """Background export loop."""
        while not self._stop_event.is_set():
            time.sleep(self.config.export_interval_sec)
            try:
                self.export_to_file()
            except Exception as e:
                if self.config.enable_debug_logging:
                    print(f"[MetricsRegistry] Export failed: {e}")


# Global singleton instance
_GLOBAL_REGISTRY: MetricsRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    """Get or create global metrics registry singleton."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _REGISTRY_LOCK:
            if _GLOBAL_REGISTRY is None:
                _GLOBAL_REGISTRY = MetricsRegistry()
                _GLOBAL_REGISTRY.start_auto_export()
    return _GLOBAL_REGISTRY


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricConfig",
    "MetricsRegistry",
    "get_metrics_registry",
]
