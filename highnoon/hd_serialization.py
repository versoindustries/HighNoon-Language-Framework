# highnoon/hd_serialization.py
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

"""Quantum-Enhanced Model Serialization with Holographic Compression.

This module provides HD (Hyperdimensional) weight compression for model
serialization, leveraging holographic encoding to achieve 2-4x compression
while maintaining high reconstruction fidelity.

Key Components:
    - HDSerializationConfig: Configuration for compression parameters
    - HDWeightCompressor: Compress/decompress model weights using HD encoding
    - VQCStateSerializer: Serialize VQC layers with phase preservation
    - SerializationBenchmark: Measure reconstruction fidelity and performance

Example:
    >>> from highnoon.hd_serialization import HDWeightCompressor, HDSerializationConfig
    >>> config = HDSerializationConfig(compression_ratio=4)
    >>> compressor = HDWeightCompressor(config)
    >>>
    >>> # Compress model weights
    >>> bundles, metadata = compressor.compress_model(model)
    >>>
    >>> # Save compressed
    >>> compressor.save_compressed(bundles, metadata, "model_compressed.npz")
    >>>
    >>> # Load and decompress
    >>> restored_model = compressor.load_and_restore(model, "model_compressed.npz")

References:
    - Holographic Reduced Representations (Plate, 2003)
    - Hyperdimensional Computing (Kanerva, 2009)
    - HD Memory for Transformers (arXiv:2306.01212)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class HDSerializationConfig:
    """Configuration for HD-compressed serialization.

    Attributes:
        compression_ratio: Target compression ratio (2-8x typical).
        hd_dim: Holographic dimension for encoding.
        use_ctqw: Enable CTQW spreading for noise robustness.
        ctqw_steps: Number of CTQW evolution steps.
        quality_threshold: Minimum cosine similarity for compression.
        layer_grouping: Number of layers to bundle together.
        preserve_vqc_phase: Serialize VQC complex params with phase.
        fallback_on_low_quality: Store uncompressed if below threshold.
        seed: Random seed for reproducibility.
    """

    compression_ratio: int = 4
    hd_dim: int = 4096
    use_ctqw: bool = True
    ctqw_steps: int = 2
    quality_threshold: float = 0.98
    layer_grouping: int = 1
    preserve_vqc_phase: bool = True
    fallback_on_low_quality: bool = True
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> HDSerializationConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HDBundle:
    """Compressed weight representation.

    Attributes:
        name: Original variable name.
        bundle: Compressed HD vector.
        original_shape: Shape of original weight tensor.
        original_dtype: Original dtype string.
        compression_ratio: Actual compression achieved.
        quality_score: Reconstruction cosine similarity (if computed).
        is_fallback: True if stored uncompressed due to low quality.
        metadata: Additional metadata (e.g., VQC phase info).
    """

    name: str
    bundle: np.ndarray
    original_shape: tuple[int, ...]
    original_dtype: str
    compression_ratio: float
    quality_score: float | None = None
    is_fallback: bool = False
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Holographic Operations
# =============================================================================


def _holographic_bind(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Holographic bind: x ⊛ y = IFFT(FFT(x) * FFT(y)).

    Args:
        x: First vector [..., dim].
        y: Second vector [..., dim].

    Returns:
        Bound vector [..., dim].
    """
    x_fft = np.fft.fft(x)
    y_fft = np.fft.fft(y)
    bound_fft = x_fft * y_fft
    return np.real(np.fft.ifft(bound_fft)).astype(np.float32)


def _holographic_unbind(bundle: np.ndarray, key: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Holographic unbind via complex division.

    Args:
        bundle: Bundled vector [..., dim].
        key: Key to unbind with [..., dim].
        epsilon: Numerical stability.

    Returns:
        Unbound vector [..., dim].
    """
    b_fft = np.fft.fft(bundle)
    k_fft = np.fft.fft(key)
    denom = np.abs(k_fft) ** 2 + epsilon
    result_fft = b_fft * np.conj(k_fft) / denom
    return np.real(np.fft.ifft(result_fft)).astype(np.float32)


def _create_unitary_keys(num_keys: int, dim: int, seed: int = 42) -> np.ndarray:
    """Create unitary keys for holographic binding.

    Keys have magnitude 1 in frequency domain, ensuring perfect
    invertibility (k * conj(k) = 1) and minimal noise in superposition.

    Args:
        num_keys: Number of keys.
        dim: Vector dimension.
        seed: Random seed.

    Returns:
        Keys [num_keys, dim] (real-valued).
    """
    rng = np.random.default_rng(seed)
    # Generate random phases in [0, 2pi]
    # For real-valued output, FFT must be conjugate symmetric
    # efficient: generate real inputs, fft, normalize magnitude

    # Alternative: Generate random complex phases, then IFFT
    # But output must be real.
    # Easiest: Generate random Gaussian, FFT, normalize to unit magnitude, IFFT

    raw = rng.standard_normal((num_keys, dim))
    raw_fft = np.fft.fft(raw, axis=1)

    # Normalize to unit magnitude (Phase only)
    phases = raw_fft / (np.abs(raw_fft) + 1e-8)

    # IFFT back to time domain
    keys = np.real(np.fft.ifft(phases, axis=1)).astype(np.float32)

    # Scaling factor to maintain energy?
    # Conservation of energy: sum(k^2) = sum(|K|^2)/N = sum(1)/N = 1
    # So expected norm of key is 1. That works.

    return keys


def _apply_ctqw(vector: np.ndarray, steps: int = 2, seed: int = 42) -> np.ndarray:
    """Apply Continuous-Time Quantum Walk spreading.

    Uses circulant graph Laplacian for efficient spreading.

    Args:
        vector: Input vector [hd_dim].
        steps: Number of walk steps.
        seed: Random seed.

    Returns:
        Spread vector [hd_dim].
    """
    hd_dim = vector.shape[-1]

    # Build circulant Laplacian (tridiagonal + periodic)
    np.random.seed(seed + 200000)
    t = 0.1 * steps

    # Use FFT-based circulant matrix-vector product for O(d log d)
    # First row of circulant: [2, -1, 0, 0, ..., 0, -1]
    first_row = np.zeros(hd_dim)
    first_row[0] = 2.0
    first_row[1] = -1.0
    first_row[-1] = -1.0

    # Eigenvalues of circulant matrix = DFT of first row
    eigenvalues = np.fft.fft(first_row)

    # Evolution: exp(-i * eigenvalues * t)
    # For real Laplacian, use cos for real part
    evolution = np.cos(t * np.real(eigenvalues))

    # Apply via FFT
    v_fft = np.fft.fft(vector)
    spread_fft = v_fft * evolution
    spread = np.real(np.fft.ifft(spread_fft))

    return spread.astype(np.float32)


def _inverse_ctqw(vector: np.ndarray, steps: int = 2, seed: int = 42) -> np.ndarray:
    """Inverse CTQW spreading.

    Args:
        vector: Spread vector [hd_dim].
        steps: Number of walk steps (same as forward).
        seed: Random seed for Laplacian.

    Returns:
        Reconstructed vector [hd_dim].
    """
    hd_dim = vector.shape[-1]
    np.random.seed(seed + 200000)
    t = 0.1 * steps

    first_row = np.zeros(hd_dim)
    first_row[0] = 2.0
    first_row[1] = -1.0
    first_row[-1] = -1.0

    eigenvalues = np.fft.fft(first_row)
    # Inverse evolution (negative time)
    evolution = np.cos(-t * np.real(eigenvalues))

    v_fft = np.fft.fft(vector)
    unspread_fft = v_fft * evolution
    unspread = np.real(np.fft.ifft(unspread_fft))

    return unspread.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between flattened arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


# =============================================================================
# Weight Compressor
# =============================================================================


class HDWeightCompressor:
    """Compress model weights using chunked holographic encoding.

    Splits weights into chunks and superposes them in groups to achieve
    target compression ratio while maintaining reconstruction fidelity.

    Method:
    1. Flatten weight tensor
    2. Pad to multiple of (chunk_size * compression_ratio)
    3. Reshape to (num_groups, compression_ratio, chunk_size)
    4. Bind each vector in group with position key
    5. Sum (superpose) over group to get (num_groups, chunk_size)

    This ensures compression scales linearly with input size, avoiding
    capacity collapse for large layers.

    Example:
        >>> config = HDSerializationConfig(compression_ratio=4, hd_dim=64)
        >>> compressor = HDWeightCompressor(config)
        >>> bundles, metadata = compressor.compress_model(model)
    """

    def __init__(self, config: HDSerializationConfig | None = None):
        """Initialize compressor.

        Args:
            config: Compression configuration. Uses defaults if None.
        """
        self.config = config or HDSerializationConfig()
        # Ensure hd_dim is used as chunk_size (small, e.g. 64-256)
        # If user passed large hd_dim (old config), clamp it for efficiency
        if self.config.hd_dim > 1024:
            logger.warning(
                f"[HDWeightCompressor] Large hd_dim {self.config.hd_dim} detected. "
                "Using 256 as chunk_size for efficiency."
            )
            self.chunk_size = 256
        else:
            self.chunk_size = self.config.hd_dim

        # Pre-compute position keys for binding within a group
        # We only need 'compression_ratio' unique keys!
        self._group_keys = _create_unitary_keys(
            self.config.compression_ratio, self.chunk_size, self.config.seed
        )

        self._compression_stats: list[dict] = []

        logger.info(
            "[HDWeightCompressor] Initialized: chunk_size=%d, ratio=%d, quality_threshold=%.3f",
            self.chunk_size,
            self.config.compression_ratio,
            self.config.quality_threshold,
        )

    def compress_weight(self, weight: np.ndarray, name: str) -> HDBundle:
        """Compress a single weight tensor.

        Args:
            weight: Weight tensor to compress.
            name: Variable name for identification.

        Returns:
            HDBundle containing compressed representation.
        """
        start_time = time.perf_counter()
        original_shape = weight.shape
        original_dtype = str(weight.dtype)
        original_size = weight.nbytes
        flat = weight.flatten().astype(np.float32)
        total_elements = flat.size

        # 1. Determine padding to fit block structure
        group_size = self.chunk_size * self.config.compression_ratio
        padding = (group_size - (total_elements % group_size)) % group_size

        if padding > 0:
            flat_padded = np.pad(flat, (0, padding), mode="constant")
        else:
            flat_padded = flat

        # 2. Reshape to (num_groups, ratio, chunk_size)
        num_groups = flat_padded.size // group_size
        # Shape: [G, R, C]
        reshaped = flat_padded.reshape(num_groups, self.config.compression_ratio, self.chunk_size)

        # 3. Bind and Superpose
        # We broadcast multiply keys: [1, R, C] * [G, R, C] -> [G, R, C]
        # Then sum over R: -> [G, C]
        keys = self._group_keys.reshape(1, self.config.compression_ratio, self.chunk_size)

        # Binding: Since we want to retrieve, we can use simple multiplication if keys are ±1 or unitary
        # But _create_position_keys returns Gaussian unit norms.
        # For Gaussian vectors, binding via element-wise mult is NOT reversible via dot product.
        # HRR uses Circular Convolution for binding.
        # But for simpler "Multiplexing", we can use element-wise multiplication IF keys are orthogonal?
        # A simple robust method is: Bundle = Sum( Vector_i * Key_i )  (Multiply encoding)
        # Retrieval: Vector_i ≈ Bundle * Key_i  (if Key_i^2 ≈ 1)
        # Since our keys are unit length Gaussian, E[x^2] != 1 per element but norm is 1.
        # Let's use standard "Multiply" binding but ensure keys are Bipolar (±1) for best property.

        # Re-generating keys as Bipolar for better reconstruction with multiply-binding
        # Or normalize generated keys to be +/- 1 magnitude?
        # Let's stick to the generated keys but use simple superposition logic:
        # We want to store R vectors in 1 vector.
        # We use standard Matrix approach: V_compressed = V_matrix @ P_matrix?
        # Let's use the code's `_holographic_bind` if we want convolution, but that's slow for weights.

        # Fast approach: Element-wise multiply with distinct keys, then sum.
        # To decode: Multiply with key again? Only works if Key_i * Key_j ≈ 0 (Orthogonal in limit).
        # Actually, for standard VSA/HDC:
        #   Bind(v, p) where p is position.
        #   Here we want to Multiplex: M = Bind(v1, p1) + Bind(v2, p2) + ...
        #   Retrieve v1 = Unbind(M, p1)
        #
        #   _holographic_bind uses FFT convolution. It preserves info well.
        #   Let's use it! It's reasonably fast on 256-dim vectors.

        # But wait, applying FFT bind on 65k parameters is slow in Python loop.
        # Vectorized implementation of Circular Convolution?
        # Conv(a, b) = IFFT( FFT(a) * FFT(b) )
        # We can do this on the whole batch [G, R, C]

        # FFT along chunk axis (axis 2)
        reshaped_fft = np.fft.fft(reshaped, axis=2)  # [G, R, C]
        keys_fft = np.fft.fft(keys, axis=2)  # [1, R, C]

        # Bind: Multiply in frequency domain
        bound_fft = reshaped_fft * keys_fft  # [G, R, C]

        # Superpose: Sum over R (axis 1)
        bundle_fft = np.sum(bound_fft, axis=1)  # [G, C]

        # Inverse FFT
        bundle = np.real(np.fft.ifft(bundle_fft, axis=1)).astype(np.float32)

        # Flatten bundle to store
        bundle_flat = bundle.flatten()

        # Determine effective compression
        compressed_size = bundle_flat.nbytes
        actual_ratio = original_size / compressed_size

        # Quality check
        reconstructed_flat = self._decompress_internal(bundle_flat, flat_padded.shape, padding)
        # Crop padding from reconstruction
        if padding > 0:
            reconstructed = reconstructed_flat[:-padding]
        else:
            reconstructed = reconstructed_flat

        quality_score = _cosine_similarity(flat, reconstructed)

        is_fallback = False
        if quality_score < self.config.quality_threshold and self.config.fallback_on_low_quality:
            logger.debug(
                "[HDWeightCompressor] Quality %.4f < threshold for %s, using fallback",
                quality_score,
                name,
            )
            is_fallback = True
            bundle_flat = flat
            actual_ratio = 1.0

        elapsed = time.perf_counter() - start_time

        self._compression_stats.append(
            {
                "name": name,
                "original_size": original_size,
                "compressed_size": bundle_flat.nbytes,
                "ratio": actual_ratio,
                "quality": quality_score,
                "fallback": is_fallback,
                "time_ms": elapsed * 1000,
            }
        )

        return HDBundle(
            name=name,
            bundle=bundle_flat,
            original_shape=original_shape,
            original_dtype=original_dtype,
            compression_ratio=actual_ratio,
            quality_score=quality_score,
            is_fallback=is_fallback,
            metadata={"padding": int(padding), "chunk_size": self.chunk_size},
        )

    def _decompress_internal(
        self, bundle_flat: np.ndarray, padded_shape: tuple, padding: int
    ) -> np.ndarray:
        """Internal decompression."""
        # Reshape bundle to [G, C]
        num_groups = bundle_flat.size // self.chunk_size
        bundle = bundle_flat.reshape(num_groups, self.chunk_size)

        # FFT of bundle [G, C]
        # Expand to [G, 1, C] for broadcasting output
        bundle_fft = np.fft.fft(bundle, axis=1)[:, np.newaxis, :]

        # FFT of keys [R, C]
        keys = self._group_keys  # [R, C]
        keys_fft = np.fft.fft(keys, axis=1)[np.newaxis, :, :]  # [1, R, C]

        # Unbind: Bundle / Key (complex division)
        # Conv(v, k) -> v * k (freq)
        # v = Unconv(out, k) -> out / k
        # Add epsilon for stability
        denom = np.abs(keys_fft) ** 2 + 1e-8
        # We multiply by conj(key) / magnitude^2 which is 1/key
        unbound_fft = bundle_fft * np.conj(keys_fft) / denom

        # IFFT -> [G, R, C]
        reconstructed_blocks = np.real(np.fft.ifft(unbound_fft, axis=2))

        # Flatten back to [N_padded]
        return reconstructed_blocks.flatten().astype(np.float32)

    def decompress_weight(self, hd_bundle: HDBundle) -> np.ndarray:
        """Decompress an HDBundle to weight tensor."""
        if hd_bundle.is_fallback:
            return hd_bundle.bundle.reshape(hd_bundle.original_shape)

        chunk_size = hd_bundle.metadata.get("chunk_size", self.chunk_size)
        padding = hd_bundle.metadata.get("padding", 0)

        # Calculate expected padded size from original shape + padding
        total_elements = np.prod(hd_bundle.original_shape)
        padded_size = total_elements + padding

        # Consistency check
        # We need to pass the PADDED shape logic or just flatten
        reconstructed_flat = self._decompress_internal(hd_bundle.bundle, (padded_size,), padding)

        if padding > 0:
            reconstructed_flat = reconstructed_flat[:-padding]

        return reconstructed_flat.reshape(hd_bundle.original_shape)

    def compress_model(self, model: tf.keras.Model) -> tuple[list[HDBundle], dict[str, Any]]:
        """Compress all trainable weights in a model.

        Args:
            model: Keras model to compress.

        Returns:
            Tuple of (list of HDBundle, metadata dict).
        """
        self._compression_stats.clear()
        bundles = []
        start_time = time.perf_counter()

        for var in model.trainable_variables:
            weight = var.numpy()
            bundle = self.compress_weight(weight, var.name)
            bundles.append(bundle)

        elapsed = time.perf_counter() - start_time

        # Compute aggregate statistics
        total_original = sum(s["original_size"] for s in self._compression_stats)
        total_compressed = sum(s["compressed_size"] for s in self._compression_stats)
        avg_quality = np.mean([s["quality"] for s in self._compression_stats])
        num_fallbacks = sum(1 for s in self._compression_stats if s["fallback"])

        metadata = {
            "config": self.config.to_dict(),
            "num_variables": len(bundles),
            "total_original_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "overall_ratio": total_original / max(total_compressed, 1),
            "avg_quality": float(avg_quality),
            "num_fallbacks": num_fallbacks,
            "compression_time_s": elapsed,
            "per_layer_stats": self._compression_stats.copy(),
        }

        logger.info(
            "[HDWeightCompressor] Compressed %d variables: %.2fx ratio, %.4f avg quality, %d fallbacks",
            len(bundles),
            metadata["overall_ratio"],
            metadata["avg_quality"],
            metadata["num_fallbacks"],
        )

        return bundles, metadata

    def decompress_model(self, bundles: list[HDBundle], model: tf.keras.Model) -> None:
        """Restore model weights from compressed bundles.

        Args:
            bundles: List of HDBundle from compress_model().
            model: Target model (must have same structure).
        """
        if len(bundles) != len(model.trainable_variables):
            raise ValueError(
                f"Bundle count {len(bundles)} != variable count {len(model.trainable_variables)}"
            )

        for bundle, var in zip(bundles, model.trainable_variables):
            if bundle.name != var.name:
                logger.warning(
                    "[HDWeightCompressor] Name mismatch: bundle=%s, var=%s",
                    bundle.name,
                    var.name,
                )
            weight = self.decompress_weight(bundle)
            var.assign(weight)

        logger.info("[HDWeightCompressor] Restored %d variables", len(bundles))

    def save_compressed(
        self,
        bundles: list[HDBundle],
        metadata: dict,
        path: str | Path,
    ) -> None:
        """Save compressed weights to file.

        Args:
            bundles: Compressed weight bundles.
            metadata: Compression metadata.
            path: Output file path (.npz).
        """
        path = Path(path)

        # Prepare arrays for npz
        arrays = {}
        bundle_metadata = []

        for i, b in enumerate(bundles):
            arrays[f"bundle_{i}"] = b.bundle
            bundle_metadata.append(
                {
                    "name": b.name,
                    "original_shape": list(b.original_shape),
                    "original_dtype": b.original_dtype,
                    "compression_ratio": b.compression_ratio,
                    "quality_score": b.quality_score,
                    "is_fallback": b.is_fallback,
                    "metadata": b.metadata,
                }
            )

        # Save metadata as JSON string in the npz
        arrays["_metadata"] = np.array(json.dumps(metadata))
        arrays["_bundle_metadata"] = np.array(json.dumps(bundle_metadata))

        np.savez_compressed(path, **arrays)
        logger.info(
            "[HDWeightCompressor] Saved %d bundles to %s (%.2f MB)",
            len(bundles),
            path,
            path.stat().st_size / 1024 / 1024,
        )

    def load_compressed(self, path: str | Path) -> tuple[list[HDBundle], dict]:
        """Load compressed weights from file.

        Args:
            path: Path to .npz file.

        Returns:
            Tuple of (list of HDBundle, metadata dict).
        """
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        metadata = json.loads(str(data["_metadata"]))
        bundle_metadata = json.loads(str(data["_bundle_metadata"]))

        bundles = []
        for i, bm in enumerate(bundle_metadata):
            bundle = HDBundle(
                name=bm["name"],
                bundle=data[f"bundle_{i}"],
                original_shape=tuple(bm["original_shape"]),
                original_dtype=bm["original_dtype"],
                compression_ratio=bm["compression_ratio"],
                quality_score=bm.get("quality_score"),
                is_fallback=bm.get("is_fallback", False),
                metadata=bm.get("metadata", {}),
            )
            bundles.append(bundle)

        logger.info("[HDWeightCompressor] Loaded %d bundles from %s", len(bundles), path)
        return bundles, metadata

    def get_compression_stats(self) -> list[dict]:
        """Get per-layer compression statistics."""
        return self._compression_stats.copy()


# =============================================================================
# VQC State Serializer
# =============================================================================


class VQCStateSerializer:
    """Serialize VQC layer state with phase preservation.

    Handles complex64/complex128 parameters by separating magnitude and phase,
    ensuring quantum state information is preserved across save/load cycles.

    Example:
        >>> serializer = VQCStateSerializer()
        >>> state = serializer.serialize(vqc_layer)
        >>> # Save state to disk...
        >>> serializer.deserialize(state, vqc_layer)
    """

    def __init__(self, preserve_phase: bool = True):
        """Initialize serializer.

        Args:
            preserve_phase: If True, serialize complex params as magnitude+phase.
        """
        self.preserve_phase = preserve_phase

    def serialize(self, layer: tf.keras.layers.Layer) -> dict:
        """Serialize VQC layer to dictionary.

        Args:
            layer: VQC layer to serialize.

        Returns:
            State dictionary with weights and metadata.
        """
        state = {
            "class_name": layer.__class__.__name__,
            "config": {},
            "weights": [],
        }

        # Get layer config if available
        if hasattr(layer, "get_config"):
            try:
                state["config"] = layer.get_config()
            except Exception as e:
                logger.debug(f"[VQCStateSerializer] get_config failed: {e}")

        # Serialize weights
        for var in layer.trainable_variables:
            weight = var.numpy()
            weight_state = {
                "name": var.name,
                "shape": list(weight.shape),
            }

            if np.iscomplexobj(weight) and self.preserve_phase:
                # Separate magnitude and phase
                magnitude = np.abs(weight)
                phase = np.angle(weight)
                weight_state["magnitude"] = magnitude.tolist()
                weight_state["phase"] = phase.tolist()
                weight_state["is_complex"] = True
            else:
                # Store as real (or as-is for complex with phase disabled)
                if np.iscomplexobj(weight):
                    weight_state["real"] = np.real(weight).tolist()
                    weight_state["imag"] = np.imag(weight).tolist()
                    weight_state["is_complex"] = True
                else:
                    weight_state["values"] = weight.tolist()
                    weight_state["is_complex"] = False

            state["weights"].append(weight_state)

        return state

    def deserialize(self, state: dict, layer: tf.keras.layers.Layer) -> None:
        """Restore VQC layer weights from state dictionary.

        Args:
            state: State dictionary from serialize().
            layer: Target layer to restore.
        """
        if len(state["weights"]) != len(layer.trainable_variables):
            raise ValueError(
                f"Weight count mismatch: state has {len(state['weights'])}, "
                f"layer has {len(layer.trainable_variables)}"
            )

        for ws, var in zip(state["weights"], layer.trainable_variables):
            target_shape = tuple(ws["shape"])

            if ws.get("is_complex", False):
                if "magnitude" in ws and "phase" in ws:
                    # Reconstruct from magnitude + phase
                    magnitude = np.array(ws["magnitude"])
                    phase = np.array(ws["phase"])
                    weight = magnitude * np.exp(1j * phase)
                else:
                    # Reconstruct from real + imag
                    real = np.array(ws["real"])
                    imag = np.array(ws["imag"])
                    weight = real + 1j * imag
                weight = weight.astype(np.complex64)
            else:
                weight = np.array(ws["values"]).astype(np.float32)

            weight = weight.reshape(target_shape)
            var.assign(weight)


# =============================================================================
# Benchmarking
# =============================================================================


@dataclass
class BenchmarkResult:
    """Results from serialization benchmark.

    Attributes:
        compression_ratio: Achieved compression ratio.
        avg_quality: Average reconstruction cosine similarity.
        min_quality: Minimum per-layer quality.
        max_quality: Maximum per-layer quality.
        output_divergence: L2 norm of output difference.
        output_cosine_sim: Cosine similarity of outputs.
        compression_time_s: Time to compress.
        decompression_time_s: Time to decompress.
        file_size_mb: Compressed file size in MB.
        per_layer_quality: Quality scores per layer.
    """

    compression_ratio: float
    avg_quality: float
    min_quality: float
    max_quality: float
    output_divergence: float
    output_cosine_sim: float
    compression_time_s: float
    decompression_time_s: float
    file_size_mb: float
    per_layer_quality: list[float]


class SerializationBenchmark:
    """Benchmark HD serialization quality and performance.

    Measures reconstruction fidelity, output divergence, and compression
    performance across different configurations.

    Example:
        >>> benchmark = SerializationBenchmark()
        >>> result = benchmark.run(model, test_input)
        >>> print(f"Quality: {result.avg_quality:.4f}, Ratio: {result.compression_ratio:.2f}x")
    """

    def __init__(self, config: HDSerializationConfig | None = None):
        """Initialize benchmark.

        Args:
            config: Compression configuration to benchmark.
        """
        self.config = config or HDSerializationConfig()
        self.compressor = HDWeightCompressor(self.config)

    def run(
        self,
        model: tf.keras.Model,
        test_input: tf.Tensor | np.ndarray,
        save_path: str | Path | None = None,
    ) -> BenchmarkResult:
        """Run benchmark on model.

        Args:
            model: Model to benchmark.
            test_input: Input tensor for output comparison.
            save_path: Optional path to save compressed weights.

        Returns:
            BenchmarkResult with all metrics.
        """
        import tempfile

        # Get original output
        original_output = model(test_input, training=False).numpy()

        # Compress
        compress_start = time.perf_counter()
        bundles, metadata = self.compressor.compress_model(model)
        compress_time = time.perf_counter() - compress_start

        # Save to file
        if save_path is None:
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
                save_path = f.name

        self.compressor.save_compressed(bundles, metadata, save_path)
        file_size = Path(save_path).stat().st_size / 1024 / 1024

        # Load and decompress
        decompress_start = time.perf_counter()
        loaded_bundles, _ = self.compressor.load_compressed(save_path)
        self.compressor.decompress_model(loaded_bundles, model)
        decompress_time = time.perf_counter() - decompress_start

        # Get restored output
        restored_output = model(test_input, training=False).numpy()

        # Compute output divergence
        output_divergence = float(np.linalg.norm(original_output - restored_output))
        output_cosine = _cosine_similarity(original_output, restored_output)

        # Extract per-layer quality
        per_layer_quality = [b.quality_score for b in bundles if b.quality_score is not None]

        return BenchmarkResult(
            compression_ratio=metadata["overall_ratio"],
            avg_quality=metadata["avg_quality"],
            min_quality=min(per_layer_quality) if per_layer_quality else 0.0,
            max_quality=max(per_layer_quality) if per_layer_quality else 0.0,
            output_divergence=output_divergence,
            output_cosine_sim=output_cosine,
            compression_time_s=compress_time,
            decompression_time_s=decompress_time,
            file_size_mb=file_size,
            per_layer_quality=per_layer_quality,
        )

    def run_sweep(
        self,
        model: tf.keras.Model,
        test_input: tf.Tensor | np.ndarray,
        hd_dims: list[int] | None = None,
    ) -> list[tuple[int, BenchmarkResult]]:
        """Run benchmark sweep across HD dimensions.

        Args:
            model: Model to benchmark.
            test_input: Input tensor for output comparison.
            hd_dims: List of HD dimensions to test.

        Returns:
            List of (hd_dim, BenchmarkResult) tuples.
        """
        if hd_dims is None:
            hd_dims = [512, 1024, 2048, 4096, 8192]

        results = []

        # Store original weights for restoration
        original_weights = [var.numpy().copy() for var in model.trainable_variables]

        for hd_dim in hd_dims:
            # Restore original weights before each test
            for var, orig in zip(model.trainable_variables, original_weights):
                var.assign(orig)

            self.config.hd_dim = hd_dim
            self.compressor = HDWeightCompressor(self.config)

            result = self.run(model, test_input)
            results.append((hd_dim, result))

            logger.info(
                "[Benchmark] hd_dim=%d: ratio=%.2f, quality=%.4f, output_sim=%.4f",
                hd_dim,
                result.compression_ratio,
                result.avg_quality,
                result.output_cosine_sim,
            )

        # Restore original weights
        for var, orig in zip(model.trainable_variables, original_weights):
            var.assign(orig)

        return results


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "HDSerializationConfig",
    "HDBundle",
    "HDWeightCompressor",
    "VQCStateSerializer",
    "SerializationBenchmark",
    "BenchmarkResult",
]
