# src/training/ewc.py
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

import logging
import os

import h5py
import tensorflow as tf

# --- Logging Setup ---
logger = logging.getLogger(__name__)


def save_ewc_state(
    cumulative_fisher: dict[str, tf.Tensor],
    cumulative_optimal_weights: dict[str, tf.Tensor],
    checkpoint_dir: str,
    dataset_name: str,
) -> None:
    """Saves the EWC Fisher information and optimal weights to an HDF5 file."""
    ewc_path = os.path.join(checkpoint_dir, f"ewc_{dataset_name}.h5")
    if not ewc_path.endswith(".h5"):
        raise ValueError(f"Invalid file extension for EWC state: {ewc_path}")
    if os.path.exists(ewc_path) and not os.access(ewc_path, os.W_OK):
        raise PermissionError(f"No write permission for {ewc_path}")
    with h5py.File(ewc_path, "w") as f:
        # MODIFIED: The term 'fisher' is kept for backward compatibility in file structure,
        # but it now stores the Hessian values from the SophiaG optimizer.
        if cumulative_fisher:
            fisher_group = f.create_group("fisher")
            for name, value in cumulative_fisher.items():
                fisher_group.create_dataset(name, data=value.numpy())
        if cumulative_optimal_weights:
            weights_group = f.create_group("weights")
            for name, value in cumulative_optimal_weights.items():
                weights_group.create_dataset(name, data=value.numpy())
    logger.info(f"Saved EWC state to {ewc_path}")


def load_ewc_state(
    checkpoint_dir: str, dataset_name: str
) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """
    Loads EWC state from an HDF5 file. This now loads the saved Hessian tensors
    which are used as the Fisher importance weights.
    """
    ewc_path = os.path.join(checkpoint_dir, f"ewc_{dataset_name}.h5")
    cumulative_fisher, cumulative_optimal_weights = {}, {}
    if os.path.exists(ewc_path):
        try:
            with h5py.File(ewc_path, "r") as f:
                if "fisher" in f:
                    for name in f["fisher"]:
                        # Load the saved Hessian as the Fisher importance
                        cumulative_fisher[name] = tf.constant(
                            f["fisher"][name][...], dtype=tf.float32
                        )
                if "weights" in f:
                    for name in f["weights"]:
                        cumulative_optimal_weights[name] = tf.constant(
                            f["weights"][name][...], dtype=tf.float32
                        )
            logger.info(f"Loaded EWC state (Hessian-as-Fisher) from {ewc_path}")
        except Exception as e:
            logger.error(f"Failed to load EWC state from {ewc_path}: {e}")
    return cumulative_fisher, cumulative_optimal_weights


# DEPRECATED: The compute_fisher function is no longer needed as the Hessian
# from the SophiaG optimizer is used directly. It has been removed.


def get_pruned_layers(model):
    """Helper function to get pruned layers from the model."""
    pruned_layers = []
    for layer in model.layers:
        if hasattr(layer, "pruning_step"):
            pruned_layers.append(layer)
        if isinstance(layer, tf.keras.Model):
            pruned_layers.extend(get_pruned_layers(layer))
    return pruned_layers
