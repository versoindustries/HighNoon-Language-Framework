# src/quantum/runtime.py
#
# Centralized runtime configuration for quantum backends, noise models,
# and mitigation strategies.

from __future__ import annotations

from highnoon import config as hsmn_config

from .device_manager import QuantumBackend, QuantumDeviceManager
from .noise import (
    AmplitudeDampingNoiseModel,
    DepolarizingNoiseModel,
    MeasurementErrorMitigator,
    PhaseDampingNoiseModel,
    ZeroNoiseExtrapolator,
)

_INITIALIZED = False


def initialize_runtime() -> QuantumDeviceManager:
    """Configures the global quantum runtime based on `src.config` settings."""
    global _INITIALIZED
    manager = QuantumDeviceManager.global_instance()

    if _INITIALIZED:
        return manager

    backend_name = hsmn_config.QUANTUM_BACKEND
    if backend_name != "default.qubit":
        try:
            manager.register_backend(QuantumBackend(name=backend_name))
        except ValueError:
            # Backend already registered; continue.
            pass
        manager.set_active_backend(backend_name)

    noise_model = _resolve_noise_model(
        hsmn_config.QUANTUM_NOISE_MODEL, hsmn_config.QUANTUM_NOISE_STRENGTH
    )
    manager.set_noise_model(noise_model)

    if len(hsmn_config.QUANTUM_ZNE_SCALES) > 1:
        manager.set_extrapolator(ZeroNoiseExtrapolator(hsmn_config.QUANTUM_ZNE_SCALES))
    else:
        manager.set_extrapolator(None)

    _INITIALIZED = True
    return manager


def _resolve_noise_model(name: str | None, strength: float):
    if not name:
        return None
    name = name.lower()
    strength = max(0.0, float(strength))
    if name == "depolarizing":
        return DepolarizingNoiseModel(base_probability=strength)
    if name in {"amplitude_damping", "damping"}:
        return AmplitudeDampingNoiseModel(base_damping=strength)
    if name in {"phase_damping", "dephasing"}:
        return PhaseDampingNoiseModel(base_dephasing=strength)
    if name in {"readout", "measurement"}:
        return MeasurementErrorMitigator(assignment_error=strength)
    if name in {"none", "off"}:
        return None
    return None
