"""
Full-spectrum telemetry schema and routing utilities for the Verso Edge-Twin.

This module materializes Phase 4 of the roadmap by defining a canonical
schema for the expanded Marlin (M710) telemetry stream, optional accelerometer
feeds, and driver diagnostics snapshots. It also provides host-side helpers to
ingest framed serial packets (via :class:`~src.runtime.edge_twin.SerialC2Link`)
while remaining agnostic to downstream consumers (MPC, Kalman filters, HSMN).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from highnoon.runtime.edge_twin import SerialC2Link, TelemetryFrame
except Exception:  # pragma: no cover - avoid circular import issues in unit tests.
    SerialC2Link = Any  # type: ignore
    TelemetryFrame = Any  # type: ignore

FULL_SPECTRUM_SCHEMA_GUID = "verso.edge_twin.m710.full_spectrum.v1"
ACCELEROMETER_SCHEMA_GUID = "verso.edge_twin.accelerometer.toolhead.v1"
DRIVER_SCHEMA_GUID = "verso.edge_twin.tmc_diagnostics.v1"
DEFAULT_PLATFORM = os.getenv("HSMN_TARGET_PLATFORM", "general").strip().lower() or "general"


class TelemetryValidationError(ValueError):
    """Raised when the incoming payload violates the expected schema."""


@dataclass(frozen=True)
class SignalDescriptor:
    key: str
    units: str
    source: str
    description: str
    optional: bool = False
    feature_flag: str | None = None
    platforms: tuple[str, ...] = ()

    def is_applicable(self, platform: str | None) -> bool:
        if self.platforms and platform:
            return platform in self.platforms
        return True


FULL_SPECTRUM_SIGNALS: tuple[SignalDescriptor, ...] = (
    SignalDescriptor("hotend_temp_c", "°C", "thermalManager", "Primary hotend thermistor reading."),
    SignalDescriptor("bed_temp_c", "°C", "thermalManager", "Heated bed thermistor reading."),
    SignalDescriptor(
        "chamber_temp_c", "°C", "thermalManager", "Enclosure temperature.", optional=True
    ),
    SignalDescriptor("vin_voltage_v", "V", "psu_monitor", "PSU Vin rail voltage."),
    SignalDescriptor("hotend_pwm", "%", "heater_pwm", "Hotend heater duty cycle."),
    SignalDescriptor("bed_pwm", "%", "heater_pwm", "Bed heater duty cycle."),
    SignalDescriptor("part_fan_pwm", "%", "fan_manager", "Part cooling fan PWM duty."),
    SignalDescriptor("hotend_fan_pwm", "%", "fan_manager", "Hotend fan PWM duty.", optional=True),
    SignalDescriptor(
        "ambient_temp_c", "°C", "aux_sensor", "Ambient temperature sensor.", optional=True
    ),
    SignalDescriptor(
        "filament_sensor",
        "state",
        "filament_monitor",
        "Filament run-out / advance state.",
        optional=True,
    ),
    SignalDescriptor(
        "motion_planner_utilization", "%", "planner", "Planner buffer fill ratio.", optional=True
    ),
    SignalDescriptor(
        "pi_cpu_temp_c",
        "°C",
        "edge_daemon",
        "Raspberry Pi CPU temperature for co-monitoring.",
        optional=True,
        platforms=("marlin_arm",),
    ),
)

TMC_DIAGNOSTIC_FIELDS = ("rms_current_ma", "stallguard", "otpw", "cs_actual")
ACCEL_AXES = ("x", "y", "z")


@dataclass
class FullSpectrumTelemetryFrame:
    schema_guid: str
    timestamp_ns: int
    metrics: dict[str, float]
    diagnostics: dict[str, dict[str, float]]
    accelerometer: dict[str, Sequence[float]]
    missing_required: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "schema": self.schema_guid,
            "ts": self.timestamp_ns,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "accelerometer": self.accelerometer,
            "missing_required": self.missing_required,
            "metadata": self.metadata,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class FullSpectrumSignalRegistry:
    def __init__(self, descriptors: Iterable[SignalDescriptor]) -> None:
        self._descriptors = {desc.key: desc for desc in descriptors}

    def sanitize_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        platform: str | None,
    ) -> tuple[dict[str, float], list[str]]:
        sanitized: dict[str, float] = {}
        missing_required: list[str] = []
        for key, descriptor in self._descriptors.items():
            if not descriptor.is_applicable(platform):
                continue
            if key not in metrics:
                if not descriptor.optional:
                    missing_required.append(key)
                continue
            value = metrics[key]
            try:
                sanitized[key] = float(value)
            except (TypeError, ValueError):
                missing_required.append(key)
        return sanitized, missing_required


FULL_SPECTRUM_REGISTRY = FullSpectrumSignalRegistry(FULL_SPECTRUM_SIGNALS)


def _sanitize_diagnostics(payload: Mapping[str, Any] | None) -> dict[str, dict[str, float]]:
    if not payload:
        return {}
    sanitized: dict[str, dict[str, float]] = {}
    for axis, fields in payload.items():
        axis_key = str(axis).upper()
        axis_metrics: dict[str, float] = {}
        for field in TMC_DIAGNOSTIC_FIELDS:
            if field in fields:
                try:
                    axis_metrics[field] = float(fields[field])
                except (TypeError, ValueError):
                    continue
        if axis_metrics:
            axis_metrics["schema"] = DRIVER_SCHEMA_GUID
            sanitized[axis_key] = axis_metrics
    return sanitized


def _sanitize_accelerometer(payload: Mapping[str, Any] | None) -> dict[str, Sequence[float]]:
    if not payload:
        return {}
    sanitized: dict[str, Sequence[float]] = {}
    for axis in ACCEL_AXES:
        samples = payload.get(axis)
        if not samples:
            continue
        if isinstance(samples, (list, tuple)):
            sanitized[axis] = tuple(float(value) for value in samples)
    if sanitized:
        sanitized["schema"] = (ACCELEROMETER_SCHEMA_GUID,)  # marker for downstream consumers.
    return sanitized


def build_full_spectrum_frame(
    packet: Mapping[str, Any],
    *,
    platform: str | None = None,
    default_timestamp_ns: int | None = None,
) -> FullSpectrumTelemetryFrame:
    """
    Convert a raw telemetry payload (from M710 or another RPC source) into a
    strongly typed :class:`FullSpectrumTelemetryFrame`.
    """

    platform = platform or DEFAULT_PLATFORM
    timestamp_ns = int(packet.get("ts") or default_timestamp_ns or time.time_ns())
    metrics_payload = packet.get("metrics") or packet
    sanitized_metrics, missing_required = FULL_SPECTRUM_REGISTRY.sanitize_metrics(
        metrics_payload, platform=platform
    )

    diagnostics_payload = packet.get("drivers")
    accelerometer_payload = packet.get("accelerometer")

    diagnostics = _sanitize_diagnostics(diagnostics_payload)
    accelerometer = _sanitize_accelerometer(accelerometer_payload)

    metadata = {
        "schema": packet.get("schema") or FULL_SPECTRUM_SCHEMA_GUID,
        "platform": platform,
        "sequence": packet.get("seq"),
        "raw_size": len(json.dumps(packet, default=str).encode("utf-8")) if packet else 0,
    }

    return FullSpectrumTelemetryFrame(
        schema_guid=FULL_SPECTRUM_SCHEMA_GUID,
        timestamp_ns=timestamp_ns,
        metrics=sanitized_metrics,
        diagnostics=diagnostics,
        accelerometer=accelerometer,
        missing_required=missing_required,
        metadata=metadata,
    )


class EdgeTelemetryRouter:
    """
    High-level helper that consumes :class:`TelemetryFrame` events from a
    :class:`SerialC2Link`, materializes the full-spectrum schema, and persists
    them to disk or downstream callbacks.
    """

    def __init__(
        self,
        *,
        serial_link: SerialC2Link,
        log_path: Path,
        platform: str | None = None,
        on_frame: Callable[[FullSpectrumTelemetryFrame], None] | None = None,
        poll_timeout: float = 0.2,
    ) -> None:
        self.serial_link = serial_link
        self.log_path = Path(log_path)
        self.platform = platform or DEFAULT_PLATFORM
        self.on_frame = on_frame
        self.poll_timeout = poll_timeout
        self._log_file = None
        self._ensure_log_file()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._ensure_log_file()
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="EdgeTelemetryRouter", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._log_file and not self._log_file.closed:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = self.serial_link.receive_frame(timeout=self.poll_timeout)
            if frame is None:
                continue
            payload = getattr(frame, "payload", None) or {}
            telemetry_payload = payload.get("telemetry") or payload.get("metrics")
            if not telemetry_payload:
                continue
            fs_frame = build_full_spectrum_frame(
                telemetry_payload,
                platform=self.platform,
                default_timestamp_ns=getattr(frame, "timestamp_ns", None),
            )
            self._persist(fs_frame)
            if self.on_frame:
                self.on_frame(fs_frame)

    def _persist(self, frame: FullSpectrumTelemetryFrame) -> None:
        if not self._log_file:
            self._ensure_log_file()
        self._log_file.write(frame.to_json() + "\n")
        self._log_file.flush()

    @property
    def _log_path(self) -> Path:
        return self.log_path

    def _ensure_log_file(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_file is None or self._log_file.closed:
            self._log_file = self.log_path.open("a", encoding="utf-8")


__all__ = [
    "ACCELEROMETER_SCHEMA_GUID",
    "ACCEL_AXES",
    "DRIVER_SCHEMA_GUID",
    "EdgeTelemetryRouter",
    "FULL_SPECTRUM_REGISTRY",
    "FULL_SPECTRUM_SCHEMA_GUID",
    "FULL_SPECTRUM_SIGNALS",
    "FullSpectrumTelemetryFrame",
    "SignalDescriptor",
    "TelemetryValidationError",
    "build_full_spectrum_frame",
]
