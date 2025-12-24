# src/quantum/__init__.py
#
# Central exports for the quantum integration utilities used by HSMN.

from .device_manager import QuantumDeviceManager
from .layers import EvolutionTimeVQCLayer, HybridVQCLayer, QuantumEnergyLayer
from .noise import (
    AmplitudeDampingNoiseModel,
    DepolarizingNoiseModel,
    MeasurementErrorMitigator,
    NoiseModel,
    PhaseDampingNoiseModel,
    ZeroNoiseExtrapolator,
)
from .qbm import QuantumBoltzmannMachine
from .qgan import QuantumMemoryAugmentor
from .runtime import initialize_runtime

__all__ = [
    "QuantumDeviceManager",
    "HybridVQCLayer",
    "EvolutionTimeVQCLayer",
    "QuantumEnergyLayer",
    "QuantumBoltzmannMachine",
    "QuantumMemoryAugmentor",
    "NoiseModel",
    "DepolarizingNoiseModel",
    "AmplitudeDampingNoiseModel",
    "PhaseDampingNoiseModel",
    "MeasurementErrorMitigator",
    "ZeroNoiseExtrapolator",
    "initialize_runtime",
]

# Configure the runtime as soon as the package is imported.
initialize_runtime()
