"""Quantum backend executors."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    from .qiskit_executor import QiskitVQCExecutor, is_qiskit_available
except ImportError:  # pragma: no cover - Qiskit not installed
    QiskitVQCExecutor = None  # type: ignore

    def is_qiskit_available() -> bool:
        return False


__all__ = [
    "QiskitVQCExecutor",
    "is_qiskit_available",
]
