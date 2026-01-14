# highnoon/training/spectral_cache.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified Spectral Cache for eigenvalue/spectrum computation sharing.

Phase 1000: Unified Spectral Cache consolidates all eigenvalue and spectral
decomposition computations across the framework to avoid redundant O(k × d²)
computations.

Consumers:
    - QULS spectral_entropy() and spectral_flatness_loss()
    - HD gradient compression spectral analysis
    - MPS bond entropy computation in UnifiedQuantumBus
    - GaLore rank selection

Memory: O(max_cache_size × k) where k = num_eigenvalues
Time: O(k × power_iter × d²) amortized via caching

Example:
    >>> from highnoon.training.spectral_cache import get_spectral_cache
    >>> cache = get_spectral_cache()
    >>> eigenvalues = cache.get_eigenvalues(hidden_states, num_eigenvalues=8)
    >>> svd_result = cache.get_svd(tensor)
    >>> cache.clear()  # Call at end of training step

Reference:
    update.md Phase 1000: Unified Spectral Cache
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import tensorflow as tf

logger = logging.getLogger(__name__)


class ComputeMode(Enum):
    """Spectral computation mode."""

    POWER_ITERATION = "power_iteration"  # O(k × power_iter × d²)
    FFT_SPECTRUM = "fft_spectrum"  # O(d log d) for HD-space
    SVD = "svd"  # O(d³) but more accurate


class SVDResult(NamedTuple):
    """Container for SVD result."""

    s: tf.Tensor  # Singular values
    u: tf.Tensor  # Left singular vectors
    v: tf.Tensor  # Right singular vectors


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    key: int
    value: tf.Tensor | SVDResult
    mode: ComputeMode
    shape_signature: tuple[int, ...]
    access_count: int = 0


class SpectralCache:
    """Unified cache for eigenvalue and spectral decomposition computations.

    This class provides thread-safe caching for expensive spectral computations
    used across multiple framework components. It implements LRU eviction and
    supports multiple computation modes.

    Attributes:
        max_cache_size: Maximum number of cached computations.
        default_num_eigenvalues: Default number of eigenvalues to compute.
        default_power_iter_steps: Default power iteration steps.

    Thread Safety:
        All public methods are thread-safe via internal locking.

    Memory Management:
        Cache should be cleared at the end of each training step via clear()
        to prevent memory leaks from tensor references.
    """

    def __init__(
        self,
        max_cache_size: int = 16,
        default_num_eigenvalues: int = 8,
        default_power_iter_steps: int = 10,
    ):
        """Initialize SpectralCache.

        Args:
            max_cache_size: Maximum number of cached entries.
            default_num_eigenvalues: Default k for top-k eigenvalue computation.
            default_power_iter_steps: Default power iteration steps.
        """
        self._eigenvalue_cache: dict[int, CacheEntry] = {}
        self._svd_cache: dict[int, CacheEntry] = {}
        self._spectrum_cache: dict[int, CacheEntry] = {}

        self._max_cache_size = max_cache_size
        self._default_num_eigenvalues = default_num_eigenvalues
        self._default_power_iter_steps = default_power_iter_steps

        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0

        logger.debug(
            "[SpectralCache] Initialized with max_size=%d, k=%d, power_iter=%d",
            max_cache_size,
            default_num_eigenvalues,
            default_power_iter_steps,
        )

    def get_eigenvalues(
        self,
        tensor: tf.Tensor,
        num_eigenvalues: int | None = None,
        power_iter_steps: int | None = None,
        epsilon: float = 1e-8,
    ) -> tf.Tensor:
        """Get cached eigenvalues or compute if not cached.

        Computes top-k eigenvalues of the covariance matrix of the input tensor
        using deflation-based power iteration.

        Complexity: O(k × power_iter × d²) on cache miss, O(1) on cache hit.

        Args:
            tensor: Input tensor [batch, seq, dim] or [batch, dim] or [N, dim].
            num_eigenvalues: Number of top eigenvalues to compute.
            power_iter_steps: Power iteration steps per eigenvalue.
            epsilon: Numerical stability constant.

        Returns:
            Top-k eigenvalues as tensor [k].

        Example:
            >>> eigenvalues = cache.get_eigenvalues(hidden_states, num_eigenvalues=8)
        """
        num_eigenvalues = num_eigenvalues or self._default_num_eigenvalues
        power_iter_steps = power_iter_steps or self._default_power_iter_steps

        cache_key = id(tensor)

        with self._lock:
            if cache_key in self._eigenvalue_cache:
                entry = self._eigenvalue_cache[cache_key]
                entry.access_count += 1
                self._hits += 1
                return entry.value

            self._misses += 1

        # Compute eigenvalues (outside lock for parallelism)
        eigenvalues = self._power_iteration_eigenvalues(
            tensor, num_eigenvalues, power_iter_steps, epsilon
        )

        with self._lock:
            self._evict_if_needed(self._eigenvalue_cache)
            self._eigenvalue_cache[cache_key] = CacheEntry(
                key=cache_key,
                value=eigenvalues,
                mode=ComputeMode.POWER_ITERATION,
                shape_signature=tuple(tensor.shape.as_list()),
            )

        return eigenvalues

    def get_svd(
        self,
        tensor: tf.Tensor,
        full_matrices: bool = False,
    ) -> SVDResult:
        """Get cached SVD or compute if not cached.

        Computes singular value decomposition: A = U @ diag(S) @ V^T

        Complexity: O(min(m,n)² × max(m,n)) on cache miss, O(1) on cache hit.

        Args:
            tensor: 2D tensor [m, n] to decompose.
            full_matrices: If True, compute full U and V matrices.

        Returns:
            SVDResult containing (s, u, v) tensors.

        Example:
            >>> result = cache.get_svd(weight_matrix)
            >>> singular_values = result.s
        """
        cache_key = id(tensor)

        with self._lock:
            if cache_key in self._svd_cache:
                entry = self._svd_cache[cache_key]
                entry.access_count += 1
                self._hits += 1
                return entry.value

            self._misses += 1

        # Compute SVD (outside lock)
        s, u, v = tf.linalg.svd(tensor, full_matrices=full_matrices)
        result = SVDResult(s=s, u=u, v=v)

        with self._lock:
            self._evict_if_needed(self._svd_cache)
            self._svd_cache[cache_key] = CacheEntry(
                key=cache_key,
                value=result,
                mode=ComputeMode.SVD,
                shape_signature=tuple(tensor.shape.as_list()),
            )

        return result

    def get_fft_spectrum(
        self,
        tensor: tf.Tensor,
        epsilon: float = 1e-8,
    ) -> tf.Tensor:
        """Get cached FFT power spectrum or compute if not cached.

        Computes power spectrum via FFT for HD-space spectral analysis.

        Complexity: O(d log d) on cache miss, O(1) on cache hit.

        Args:
            tensor: Input tensor to analyze.
            epsilon: Numerical stability constant.

        Returns:
            Power spectrum tensor.

        Example:
            >>> spectrum = cache.get_fft_spectrum(hd_embedding)
        """
        cache_key = id(tensor)

        with self._lock:
            if cache_key in self._spectrum_cache:
                entry = self._spectrum_cache[cache_key]
                entry.access_count += 1
                self._hits += 1
                return entry.value

            self._misses += 1

        # Compute FFT power spectrum (outside lock)
        spectrum = self._compute_fft_spectrum(tensor, epsilon)

        with self._lock:
            self._evict_if_needed(self._spectrum_cache)
            self._spectrum_cache[cache_key] = CacheEntry(
                key=cache_key,
                value=spectrum,
                mode=ComputeMode.FFT_SPECTRUM,
                shape_signature=tuple(tensor.shape.as_list()),
            )

        return spectrum

    def get_spectral_entropy(
        self,
        tensor: tf.Tensor,
        num_eigenvalues: int | None = None,
        power_iter_steps: int | None = None,
        epsilon: float = 1e-8,
    ) -> tf.Tensor:
        """Compute spectral entropy from cached eigenvalues.

        Convenience method that computes normalized Von Neumann entropy
        from the eigenvalue distribution.

        Args:
            tensor: Input tensor.
            num_eigenvalues: Number of eigenvalues.
            power_iter_steps: Power iteration steps.
            epsilon: Numerical stability constant.

        Returns:
            Normalized spectral entropy in [0, 1].
        """
        eigenvalues = self.get_eigenvalues(tensor, num_eigenvalues, power_iter_steps, epsilon)

        # Normalize eigenvalues to probability
        eigenvalues = tf.maximum(eigenvalues, epsilon)
        total = tf.reduce_sum(eigenvalues)
        probs = eigenvalues / (total + epsilon)

        # Compute entropy: H = -Σ p log(p)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + epsilon))

        # Normalize by log(k) to get value in [0, 1]
        k = tf.cast(tf.shape(eigenvalues)[0], tf.float32)
        max_entropy = tf.math.log(k)
        normalized_entropy = entropy / (max_entropy + epsilon)

        return normalized_entropy

    def get_spectral_flatness(
        self,
        tensor: tf.Tensor,
        num_eigenvalues: int | None = None,
        power_iter_steps: int | None = None,
        epsilon: float = 1e-8,
    ) -> tf.Tensor:
        """Compute spectral flatness from cached eigenvalues.

        Spectral flatness = geometric_mean / arithmetic_mean.
        Values close to 1 indicate flat spectrum, close to 0 indicate peaked.

        Args:
            tensor: Input tensor.
            num_eigenvalues: Number of eigenvalues.
            power_iter_steps: Power iteration steps.
            epsilon: Numerical stability constant.

        Returns:
            Spectral flatness in [0, 1].
        """
        eigenvalues = self.get_eigenvalues(tensor, num_eigenvalues, power_iter_steps, epsilon)

        eigenvalues = tf.maximum(eigenvalues, epsilon)

        # Compute geometric mean via log
        log_eigenvalues = tf.math.log(eigenvalues)
        log_geometric_mean = tf.reduce_mean(log_eigenvalues)
        geometric_mean = tf.exp(log_geometric_mean)

        # Compute arithmetic mean
        arithmetic_mean = tf.reduce_mean(eigenvalues)

        flatness = geometric_mean / (arithmetic_mean + epsilon)
        flatness = tf.minimum(flatness, 1.0)

        return flatness

    def clear(self) -> None:
        """Clear all cached computations.

        Should be called at the end of each training step to release tensor
        references and prevent memory leaks.
        """
        with self._lock:
            self._eigenvalue_cache.clear()
            self._svd_cache.clear()
            self._spectrum_cache.clear()
            logger.debug(
                "[SpectralCache] Cleared. Stats: hits=%d, misses=%d, hit_rate=%.2f%%",
                self._hits,
                self._misses,
                100 * self._hits / max(self._hits + self._misses, 1),
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(total, 1),
                "eigenvalue_cache_size": len(self._eigenvalue_cache),
                "svd_cache_size": len(self._svd_cache),
                "spectrum_cache_size": len(self._spectrum_cache),
                "max_cache_size": self._max_cache_size,
            }

    def _evict_if_needed(self, cache: dict[int, CacheEntry]) -> None:
        """Evict oldest entry if cache is full (LRU policy)."""
        if len(cache) >= self._max_cache_size:
            # Find entry with lowest access count
            oldest_key = min(cache.keys(), key=lambda k: cache[k].access_count)
            del cache[oldest_key]

    def _power_iteration_eigenvalues(
        self,
        tensor: tf.Tensor,
        num_eigenvalues: int,
        power_iter_steps: int,
        epsilon: float,
    ) -> tf.Tensor:
        """Compute top-k eigenvalues via deflation-based power iteration.

        Complexity: O(k × power_iter × d²)

        Args:
            tensor: Input tensor.
            num_eigenvalues: Number of eigenvalues k.
            power_iter_steps: Iterations per eigenvalue.
            epsilon: Numerical stability constant.

        Returns:
            Top-k eigenvalues [k].
        """
        h = tf.cast(tensor, tf.float32)

        # Flatten to [N, dim]
        if len(h.shape) == 3:
            h_flat = tf.reshape(h, [-1, h.shape[-1]])
        elif len(h.shape) == 1:
            h_flat = tf.expand_dims(h, 0)
        else:
            h_flat = h

        # Center the data
        mean = tf.reduce_mean(h_flat, axis=0, keepdims=True)
        h_centered = h_flat - mean

        # Compute covariance: (1/N) * H^T * H
        n_samples = tf.cast(tf.shape(h_centered)[0], tf.float32)
        cov = tf.matmul(h_centered, h_centered, transpose_a=True) / n_samples

        # Power iteration with deflation
        dim = tf.shape(cov)[0]
        k = tf.minimum(num_eigenvalues, dim)
        eigenvalues = []

        A = cov
        for _ in range(k):
            # Random initialization
            v = tf.random.normal([dim, 1], dtype=tf.float32)
            # GRADIENT FIX: Add epsilon to prevent NaN/Inf gradients when norm → 0
            v = v / (tf.norm(v) + 1e-8)

            # Power iteration
            for _ in range(power_iter_steps):
                v_new = tf.matmul(A, v)
                v_new = v_new / (tf.norm(v_new) + epsilon)
                v = v_new

            # Extract eigenvalue: λ = v^T A v
            eigenvalue = tf.squeeze(tf.matmul(tf.matmul(v, A, transpose_a=True), v))
            eigenvalue = tf.maximum(eigenvalue, epsilon)
            eigenvalues.append(eigenvalue)

            # Deflate: A = A - λ * v * v^T
            A = A - eigenvalue * tf.matmul(v, v, transpose_b=True)

        return tf.stack(eigenvalues)

    def _compute_fft_spectrum(
        self,
        tensor: tf.Tensor,
        epsilon: float,
    ) -> tf.Tensor:
        """Compute FFT power spectrum.

        Complexity: O(d log d)

        Args:
            tensor: Input tensor.
            epsilon: Numerical stability constant.

        Returns:
            Power spectrum.
        """
        h = tf.cast(tensor, tf.float32)

        # Flatten if needed
        if len(h.shape) > 2:
            h = tf.reshape(h, [-1, h.shape[-1]])
        elif len(h.shape) == 1:
            h = tf.expand_dims(h, 0)

        # Phase 1.5: Compute FFT with complex128 precision per GRADIENT_CONNECTIVITY_ROADMAP
        h_complex = tf.cast(h, tf.complex128)
        fft = tf.signal.fft(h_complex)

        # Power spectrum: |FFT|²
        power = tf.cast(tf.abs(fft) ** 2, tf.float32)

        # Average over samples
        spectrum = tf.reduce_mean(power, axis=0)

        # Normalize
        spectrum = spectrum / (tf.reduce_sum(spectrum) + epsilon)

        return spectrum


# =============================================================================
# Global Instance (Singleton Pattern)
# =============================================================================

_global_spectral_cache: SpectralCache | None = None
_global_cache_lock = threading.Lock()


def get_spectral_cache() -> SpectralCache:
    """Get the global SpectralCache instance.

    Returns the singleton SpectralCache instance, creating it if necessary.
    Thread-safe.

    Returns:
        Global SpectralCache instance.

    Example:
        >>> cache = get_spectral_cache()
        >>> eigenvalues = cache.get_eigenvalues(hidden_states)
    """
    global _global_spectral_cache

    if _global_spectral_cache is None:
        with _global_cache_lock:
            if _global_spectral_cache is None:
                _global_spectral_cache = SpectralCache()
                logger.info("[SpectralCache] Global instance created")

    return _global_spectral_cache


def clear_spectral_cache() -> None:
    """Clear the global spectral cache.

    Should be called at the end of each training step.
    """
    cache = get_spectral_cache()
    cache.clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SpectralCache",
    "SVDResult",
    "ComputeMode",
    "get_spectral_cache",
    "clear_spectral_cache",
]
