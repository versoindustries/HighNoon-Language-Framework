import logging
import sys

import numpy as np
import tensorflow as tf

from highnoon import config
from highnoon._native.ops.meta_controller_op import trigger_meta_controller

logger = logging.getLogger(__name__)


class HamiltonianMetaControllerCallback(tf.keras.callbacks.Callback):
    """
    A Keras Callback to orchestrate the Hamiltonian Meta-Controller.

    This callback is manually triggered from the custom training loop. At the end
    of each batch (or every N batches), it collects metrics from the `logs`
    dictionary, formats them into dynamic tensors, and calls the C++ meta-controller op.

    Phase 400: Includes HD-based stagnation detection via metric fingerprinting.
    """

    def __init__(self, frequency=10, trigger_sysid_reload: bool = False):
        """
        Initializes the callback.
        Args:
            frequency (int): How often (in batches) to trigger the controller.
            trigger_sysid_reload (bool): If True, signals the C++ controller to
                                         reload its config files on the first call.
        """
        super().__init__()
        self.frequency = frequency
        self.trigger_sysid_reload = tf.constant(trigger_sysid_reload, dtype=tf.bool)
        self._reload_triggered = False  # Ensures the reload signal is only sent once.

        # Phase 400: HD Stagnation Detection
        self._use_hd_fingerprinting = config.USE_HD_CONTROL_FINGERPRINTING
        self._hd_fingerprint_dim = config.HD_CONTROL_FINGERPRINT_DIM
        self._stagnation_threshold = config.HD_STAGNATION_THRESHOLD
        self._prev_fingerprint = None
        self._stagnation_similarity = 0.0
        self._stagnation_count = 0

        logger.info(
            f"HamiltonianMetaControllerCallback initialized with frequency {self.frequency}, "
            f"HD fingerprinting: {self._use_hd_fingerprinting}"
        )

    def on_batch_end(
        self,
        batch: int,
        logs: dict = None,
        force_reload: bool = False,
        trial_dir: str = None,
        control_input_names: list[str] = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Triggers the C++ meta-controller with the latest training metrics.

        This method now returns the raw output tensors from the C++ op:
        a tensor of block names and a tensor of new evolution times.
        """
        # Determine if this is a batch where the controller should be triggered based on frequency.
        should_trigger_now = (batch + 1) % self.frequency == 0
        # Determine if a reload should happen now (either one-time initial or forced by the loop).
        # --- START: DEFINITIVE FIX ---
        should_reload_now = force_reload or (
            self.trigger_sysid_reload and not self._reload_triggered
        )

        if not should_trigger_now and not should_reload_now:
            # Return a special value to indicate no update
            return tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.float32)

        tf.print("[MetaControllerCallback] Triggering C++ controller. Batch:", batch)

        logs = logs or {}
        # --- START: DEFINITIVE FIX for Empty Logs ---
        # If for some reason the logs dictionary is empty, do not call the controller.
        if not logs:
            tf.print(
                "[MetaControllerCallback] Warning: Logs dictionary is empty. Skipping controller call.",
                output_stream=sys.stderr,
            )
            return tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.float32)
        metric_values: list[tf.Tensor] = []
        metric_names: list[str] = []

        # --- START: DEFINITIVE FIX (REVERTED FAULTY FIX) ---
        # The previous logic was too restrictive and filtered out necessary metrics.
        # This corrected logic iterates through all items in the logs dictionary
        # and adds them to the lists to be sent to the C++ controller, restoring
        # the full system state visibility.
        for name, value in logs.items():
            value_tensor = tf.cast(value, tf.float32)
            is_finite = tf.reduce_all(tf.math.is_finite(value_tensor))
            if not bool(is_finite.numpy()):
                tf.print(
                    "[MetaControllerCallback] Skipping metric",
                    name,
                    "due to non-finite value.",
                    output_stream=sys.stderr,
                )
                continue
            metric_names.append(name)
            metric_values.append(value_tensor)
        # --- END: DEFINITIVE FIX ---

        tf.print(
            "[MetaControllerCallback] Sending metrics to C++:",
            metric_names,
            output_stream=sys.stderr,
        )

        # If this is the one-time reload batch, set the flag.
        if should_reload_now:
            tf.print("  -> Signaling System ID config reload.", output_stream=sys.stderr)
            # Mark the initial reload as done if it was the one that triggered this.
            self._reload_triggered = True

        # Call the C++ op wrapper
        block_names_tensor, evolution_times_tensor = trigger_meta_controller(
            metric_values=tf.stack(metric_values),
            metric_names=tf.constant(metric_names, dtype=tf.string),
            control_input_names=tf.constant(control_input_names or [], dtype=tf.string),
            trigger_autotune=tf.constant(False, dtype=tf.bool),
            trigger_system_id=should_reload_now,
            config_path=tf.constant(
                trial_dir if trial_dir else "", dtype=tf.string
            ),  # Pass trial dir
        )

        # Phase 400: HD Fingerprinting for stagnation detection
        if self._use_hd_fingerprinting and metric_values:
            self._update_hd_fingerprint(metric_values)

        return block_names_tensor, evolution_times_tensor

    def _update_hd_fingerprint(self, metric_values: list[tf.Tensor]) -> None:
        """Compute HD fingerprint from metric values and track stagnation.

        Uses random projection to compress metrics into a fingerprint vector,
        then computes cosine similarity with previous fingerprint to detect
        training stagnation (barren plateaus).
        """
        # Flatten all metrics into a single vector
        try:
            flat_metrics = tf.concat([tf.reshape(v, [-1]) for v in metric_values], axis=0)
            metric_vec = flat_metrics.numpy().astype(np.float32)
        except Exception:
            return  # Skip if conversion fails

        # Generate deterministic random projection (seeded for reproducibility)
        np.random.seed(42)
        if len(metric_vec) > 0:
            proj_matrix = np.random.randn(len(metric_vec), self._hd_fingerprint_dim).astype(
                np.float32
            )
            proj_matrix /= np.sqrt(self._hd_fingerprint_dim)

            # Compute fingerprint via random projection
            fingerprint = metric_vec @ proj_matrix

            # Normalize fingerprint
            norm = np.linalg.norm(fingerprint)
            if norm > 1e-12:
                fingerprint = fingerprint / norm

            # Compute similarity with previous fingerprint
            if self._prev_fingerprint is not None:
                dot = np.dot(fingerprint, self._prev_fingerprint)
                self._stagnation_similarity = float(np.clip(dot, 0.0, 1.0))

                # Track consecutive stagnation
                if self._stagnation_similarity > self._stagnation_threshold:
                    self._stagnation_count += 1
                    if self._stagnation_count >= 3:
                        tf.print(
                            f"[MetaControllerCallback] HD stagnation detected! "
                            f"Similarity: {self._stagnation_similarity:.4f}, "
                            f"Count: {self._stagnation_count}",
                            output_stream=sys.stderr,
                        )
                else:
                    self._stagnation_count = 0

            self._prev_fingerprint = fingerprint

    @property
    def stagnation_metric(self) -> float:
        """Get current HD stagnation similarity.

        Returns:
            Cosine similarity (0-1) between consecutive metric fingerprints.
            High values (>0.95) indicate training stagnation.
        """
        return self._stagnation_similarity

    @property
    def is_stagnating(self) -> bool:
        """Check if currently in stagnation state.

        Returns:
            True if consecutive stagnation count exceeds threshold.
        """
        return self._stagnation_count >= 3
