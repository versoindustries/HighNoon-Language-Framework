# src/quantum/device_manager.py
#
# Lightweight device and backend registry for quantum components embedded
# within the HSMN architecture.

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from highnoon.quantum.executors import QiskitVQCExecutor, is_qiskit_available

logger = logging.getLogger(__name__)


@dataclass
class QuantumBackend:
    """
    Describes a quantum execution backend.

    Attributes:
        name: Friendly identifier for the backend.
        num_qubits: Maximum number of qubits supported by the backend.
        is_hardware: True if the backend represents physical hardware.
        metadata: Optional dictionary with provider-specific information.
    """

    name: str
    num_qubits: int = 1
    is_hardware: bool = False
    metadata: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "is_hardware": self.is_hardware,
            "metadata": self.metadata or {},
        }


class StatevectorBackend(QuantumBackend):
    """
    Default CPU-based statevector simulator.
    """

    def __init__(self, num_qubits: int = 1):
        super().__init__(
            name="default.qubit",
            num_qubits=num_qubits,
            is_hardware=False,
            metadata={"provider": "internal", "supports_batch": True},
        )


class QuantumDeviceManager:
    """
    Registry and runtime selector for quantum backends and noise models.

    The manager behaves like a singleton to make it easy for disparate modules
    to coordinate on the active quantum execution target.
    """

    _instance: QuantumDeviceManager | None = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._backends: dict[str, QuantumBackend] = {}
        self._active_backend: str | None = None
        self._noise_model = None
        self._zne_extrapolator = None
        self._executors: dict[str, Any] = {}

        # Register the default simulator immediately so the system always has a backend.
        self.register_backend(StatevectorBackend())
        self._auto_register_qiskit_backend()

    # ------------------------------------------------------------------
    # Backend management
    # ------------------------------------------------------------------
    def register_backend(self, backend: QuantumBackend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"Backend '{backend.name}' already registered.")
        self._backends[backend.name] = backend
        if self._active_backend is None:
            self._active_backend = backend.name

    def set_active_backend(self, backend_name: str) -> None:
        if backend_name not in self._backends:
            raise KeyError(f"Backend '{backend_name}' is not registered.")
        self._active_backend = backend_name

    def get_backend(self, backend_name: str | None = None) -> QuantumBackend:
        name = backend_name or self._active_backend
        if name is None:
            raise RuntimeError("No quantum backend has been configured.")
        return self._backends[name]

    def list_backends(self) -> dict[str, dict[str, Any]]:
        return {name: backend.summary() for name, backend in self._backends.items()}

    def register_executor(
        self,
        backend_name: str,
        executor: Any,
        make_default: bool = False,
    ) -> None:
        """
        Registers an execution interface for a backend. The executor must expose
        a `run_expectation` method that mirrors the signature used by
        HybridVQCLayer.
        """
        self._executors[backend_name] = executor
        if make_default:
            if backend_name not in self._backends:
                self.register_backend(QuantumBackend(name=backend_name))
            self._active_backend = backend_name

    def _auto_register_qiskit_backend(self) -> None:
        """
        Registers a Qiskit-backed executor when the dependency is available.
        """
        if not is_qiskit_available():
            return
        backend_name = "qiskit.density_matrix"
        if backend_name in self._executors:
            return
        try:
            self.enable_qiskit_backend(
                backend_name=backend_name,
                num_qubits=self._backends["default.qubit"].num_qubits,
                noise_model=self._noise_model,
                default_shots=4096,
                make_default=False,
            )
        except Exception as err:  # pragma: no cover - defensive logging path
            logger.warning("Failed to initialise Qiskit executor: %s", err)
            self._executors.pop(backend_name, None)
            if backend_name in self._backends and backend_name != self._active_backend:
                self._backends.pop(backend_name, None)

    def get_executor(self, backend_name: str | None = None):
        name = backend_name or self._active_backend
        if name is None:
            return None
        return self._executors.get(name)

    def enable_qiskit_backend(
        self,
        *,
        backend_name: str = "qiskit.density_matrix",
        num_qubits: int | None = None,
        noise_model: object | None = None,
        default_shots: int | None = 4096,
        make_default: bool = True,
    ) -> None:
        """
        Registers (or refreshes) the Qiskit executor with optional noise modelling.
        """
        if not is_qiskit_available():
            raise ImportError(
                "Qiskit is not installed. Install the optional dependencies listed in requirements.txt."
            )

        target_qubits = (
            int(num_qubits)
            if num_qubits is not None
            else self._backends.get("default.qubit", QuantumBackend("default.qubit")).num_qubits
        )

        if backend_name not in self._backends:
            qiskit_backend = QuantumBackend(
                name=backend_name,
                num_qubits=target_qubits,
                is_hardware=False,
                metadata={
                    "provider": "qiskit",
                    "supports_noise": True,
                    "default_shots": default_shots,
                },
            )
            self.register_backend(qiskit_backend)
        else:
            self._backends[backend_name].num_qubits = target_qubits

        executor = QiskitVQCExecutor(
            num_qubits=target_qubits,
            noise_model=noise_model if noise_model is not None else self._noise_model,
            default_shots=default_shots,
        )
        self.register_executor(backend_name, executor, make_default=make_default)

        if noise_model is not None:
            self.set_noise_model(noise_model)
        elif make_default:
            # Ensure the active backend has the latest global noise model.
            executor.update_noise_model(self._noise_model)

    # ------------------------------------------------------------------
    # Noise model management
    # ------------------------------------------------------------------
    def set_noise_model(self, noise_model) -> None:
        """
        Assigns a noise model that all hybrid quantum components will use.
        The object must expose an `apply(expectation, scale)` method.
        """
        self._noise_model = noise_model
        for executor in self._executors.values():
            updater = getattr(executor, "update_noise_model", None)
            if updater is not None:
                updater(noise_model)

    def get_noise_model(self):
        return self._noise_model

    def apply_noise(self, expectation, scale: float = 1.0):
        if self._noise_model is None:
            return expectation
        return self._noise_model.apply(expectation, scale=scale)

    # ------------------------------------------------------------------
    # Zero-noise extrapolation management
    # ------------------------------------------------------------------
    def set_extrapolator(self, extrapolator) -> None:
        """
        Registers a ZeroNoiseExtrapolator-like utility. The object must expose
        an `extrapolate(noisy_values)` method.
        """
        self._zne_extrapolator = extrapolator

    def get_extrapolator(self):
        return self._zne_extrapolator

    # ------------------------------------------------------------------
    # Singleton helpers
    # ------------------------------------------------------------------
    @classmethod
    def global_instance(cls) -> QuantumDeviceManager:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @classmethod
    def configure_global(cls, **kwargs) -> QuantumDeviceManager:
        """
        Convenience helper to configure the global instance and return it.
        """
        manager = cls.global_instance()
        if "backend" in kwargs:
            manager.set_active_backend(kwargs["backend"])
        if "noise_model" in kwargs:
            manager.set_noise_model(kwargs["noise_model"])
        if "extrapolator" in kwargs:
            manager.set_extrapolator(kwargs["extrapolator"])
        return manager
