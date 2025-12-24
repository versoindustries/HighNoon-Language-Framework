import logging
import sys

import tensorflow as tf

from highnoon._native.ops.meta_controller_op import trigger_meta_controller

logger = logging.getLogger(__name__)


class HamiltonianMetaControllerCallback(tf.keras.callbacks.Callback):
    """
    A Keras Callback to orchestrate the Hamiltonian Meta-Controller.

    This callback is manually triggered from the custom training loop. At the end
    of each batch (or every N batches), it collects metrics from the `logs`
    dictionary, formats them into dynamic tensors, and calls the C++ meta-controller op.
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
        logger.info(
            f"HamiltonianMetaControllerCallback initialized with frequency {self.frequency}."
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

        return block_names_tensor, evolution_times_tensor
