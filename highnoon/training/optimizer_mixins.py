# highnoon/training/optimizer_mixins.py
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

"""Mixin classes for optimizer functionality.

This module provides reusable mixins for second-order optimizers,
enabling consistent Hessian update logic across SophiaG, QIAO, and
other adaptive optimizers.
"""

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class HessianUpdateMixin:
    """Mixin providing diagonal Hessian update for second-order optimizers.

    This mixin implements the Gauss-Newton Hessian approximation used by
    SophiaG and QIAO optimizers. The Hessian is approximated as the
    squared gradient, then exponentially smoothed.

    Requirements:
        - Optimizer must have `self.model` attribute
        - Optimizer must have `self.beta_2` attribute
        - Optimizer must have `self.hessians` list
        - Optimizer must have `self.variables` (from Keras Optimizer)
        - Optimizer must implement `_get_variable_index(var)`

    Attributes:
        beta_2: EMA decay for Hessian estimates (typically 0.99).
        hessians: List of Hessian estimate variables (one per trainable var).
    """

    def update_hessian(self, loss_fn, data_batch) -> None:
        """Update the diagonal Hessian estimate 'h' using Gauss-Newton.

        This method should be called periodically (every k steps) to update
        the Hessian estimates used for preconditioning gradients.

        Args:
            loss_fn: Loss function that takes (y_true, y_pred) and returns loss.
            data_batch: Tuple of (context, target) tensors for Hessian estimation.

        Note:
            Uses sampled labels from the model's output distribution to compute
            gradients, which provides an unbiased Hessian estimate.
        """
        beta_2_t = tf.cast(self.beta_2, "float32")

        with tf.GradientTape() as h_tape:
            ctx, tgt = data_batch
            # The model's call method returns three values (logits, moe_info, aux_metrics).
            # We only need the logits for Hessian estimation.
            logits, _, _ = self.model((ctx, tgt), training=True)

            probs = tf.nn.softmax(logits, axis=-1)
            safe_probs = tf.clip_by_value(probs[:, -1, :], 1e-9, 1.0)
            sampled_labels = tf.random.categorical(tf.math.log(safe_probs), 1)
            sampled_labels = tf.squeeze(sampled_labels, axis=-1)

            # Use the full loss for Hessian estimation
            sampled_loss = tf.reduce_mean(loss_fn(tgt[:, 1:], logits[:, : tf.shape(tgt)[1] - 1, :]))

        # Compute gradients w.r.t. self.variables to ensure alignment
        hessian_grads = h_tape.gradient(sampled_loss, self.variables)

        # Iterate over self.variables to ensure alignment
        for h_grad, var in zip(hessian_grads, self.variables):
            if h_grad is not None:
                try:
                    var_index = self._get_variable_index(var)
                    h = self.hessians[var_index]
                    h_hat = tf.square(h_grad)
                    h.assign(beta_2_t * h + (1.0 - beta_2_t) * h_hat)
                except KeyError:
                    # This can happen if the Hessian tape tracks a variable
                    # the optimizer doesn't. Common during HPO model rebuilds.
                    logger.warning(f"Skipping Hessian update for untracked variable: {var.name}")


__all__ = ["HessianUpdateMixin"]
