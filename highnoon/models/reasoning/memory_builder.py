# src/models/reasoning/memory_builder.py
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

import sys

import tensorflow as tf

from highnoon import config
from highnoon.quantum.qgan import QuantumMemoryAugmentor


class MemoryHierarchyBuilder(tf.keras.layers.Layer):
    """
    A layer responsible for building and processing the memory hierarchy.
    It takes an initial memory level and progressively aggregates it, then
    stacks the levels into a single memory sequence.

    MODIFIED: The internal loop now assumes that `self.graph_learner()`
    performs the full end-to-end graph processing, including aggregation, pooling,
    and projection to the next level of the hierarchy, via the Fused C++ kernel.
    The dependency on `self.aggregator`'s pooling function is removed.
    """

    def __init__(
        self,
        decompressor: tf.keras.layers.Layer,
        graph_learner: tf.keras.layers.Layer,
        level_embeddings: tf.keras.layers.Layer,
        compressed_dim: int,
        embedding_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decompressor = decompressor
        self.graph_learner = graph_learner
        self.level_embeddings = level_embeddings
        self.compressed_dim = compressed_dim
        self.embedding_dim = embedding_dim
        self.dimension_mismatch = False  # Will be set in build() if dimensions don't match
        self.memory_augmentor = None
        if config.QGAN_SAMPLES_PER_BATCH > 0:
            augmentor_name = f"{self.name}_qgan" if hasattr(self, "name") else "memory_qgan"
            self.memory_augmentor = QuantumMemoryAugmentor(
                compressed_dim=self.compressed_dim,
                samples_per_batch=config.QGAN_SAMPLES_PER_BATCH,
                num_qubits=config.QUANTUM_ENERGY_QUBITS,
                name=augmentor_name,
            )

    def build(self, input_shape):
        """
        Explicitly builds the weights for the sub-layers to prevent Keras warnings
        with complex control flow like tf.while_loop.

        ENTERPRISE FIX: Handles dimension mismatches between input and graph_learner.
        """
        # Check if graph_learner can handle the input dimensions
        # For molecular mode: input is compressed_dim, but graph_learner expects embedding_dim
        # Handle both TensorShape and tuple inputs
        if isinstance(input_shape, (tuple, list)):
            # Extract the last dimension, handling nested tuples
            last_dim = input_shape[-1]
            # If last_dim is itself a tuple/TensorShape, extract its integer value
            if isinstance(last_dim, (tuple, list, tf.TensorShape)):
                input_dim = int(last_dim[-1]) if hasattr(last_dim, "__getitem__") else int(last_dim)
            else:
                input_dim = int(last_dim) if last_dim is not None else None
        elif isinstance(input_shape, tf.TensorShape):
            input_dim = int(input_shape[-1]) if input_shape[-1] is not None else None
        else:
            input_dim = int(input_shape)

        # Detect if we have a dimension mismatch
        # GAQuantumGraphLearner has embedding_dim attribute
        graph_learner_dim = getattr(self.graph_learner, "embedding_dim", None)
        # Use compressed_dim as the authoritative input width when unknown or mismatched.
        if input_dim is None or input_dim <= 0:
            input_dim = self.compressed_dim
        self.dimension_mismatch = graph_learner_dim is not None and graph_learner_dim != input_dim

        if self.dimension_mismatch:
            input_dim = self.compressed_dim  # enforce projection input width to compressed_dim
            # Create projection layers to handle dimension mismatch
            # This allows molecular (compressed) and language (full) modes to coexist
            self.project_to_graph = tf.keras.layers.Dense(
                graph_learner_dim, name=f"{self.name}_project_to_graph"
            )
            self.project_from_graph = tf.keras.layers.Dense(
                input_dim, name=f"{self.name}_project_from_graph"
            )

            # Build projection layers
            self.project_to_graph.build(tf.TensorShape([None, None, input_dim]))
            self.project_from_graph.build(tf.TensorShape([None, None, graph_learner_dim]))

            # Build graph_learner with its expected dimensions
            if not self.graph_learner.built:
                # Handle both static and dynamic dimensions
                batch_dim = (
                    input_shape[0] if isinstance(input_shape[0], (int, type(None))) else None
                )
                seq_dim = input_shape[1] if isinstance(input_shape[1], (int, type(None))) else None
                expected_shape = tf.TensorShape([batch_dim, seq_dim, graph_learner_dim])
                self.graph_learner.build(expected_shape)
        else:
            # No dimension mismatch - build directly
            if not self.graph_learner.built:
                self.graph_learner.build(input_shape)

        if not self.decompressor.built:
            # Decompressor takes compressed_dim as input and outputs embedding_dim
            # It was configured with input_dims = compressed_dim factorization
            # So we need to build with the input dimension it expects
            self.decompressor.build(tf.TensorShape([None, self.decompressor.input_dim]))

        if not self.level_embeddings.built:
            self.level_embeddings.build(tf.TensorShape([]))

        if self.memory_augmentor is not None and not self.memory_augmentor.built:
            self.memory_augmentor.build(tf.TensorShape([None, self.compressed_dim]))

        super().build(input_shape)

    def _build_memory_hierarchy(
        self,
        initial_memory_level: tf.Tensor,
        initial_structural_level: tf.Tensor,
        nodes_per_chunk: tf.Tensor,
        training: bool = False,
        return_aux_metrics: bool = False,
    ) -> tuple[tf.TensorArray, tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """Builds the full memory hierarchy using a tf.while_loop."""
        ta = tf.TensorArray(
            dtype=tf.float32,
            size=0,
            dynamic_size=True,
            element_shape=tf.TensorShape([None, None, self.compressed_dim]),
            infer_shape=False,
            name="memory_levels_ta",
        )

        # TensorArrays to store metric keys and values from each loop iteration
        metric_keys_ta = tf.TensorArray(
            dtype=tf.string, size=0, dynamic_size=True, clear_after_read=False
        )
        metric_values_ta = tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )
        metric_idx = tf.constant(0)

        # TensorArray to accumulate QGAN losses from within the while loop
        # These losses will be summed and added after the loop completes
        qgan_losses_ta = tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )
        qgan_loss_idx = tf.constant(0)

        # Augment the initial memory level with synthetic samples when training.
        current_level_raw = initial_memory_level
        current_level_augmented = current_level_raw
        current_structural_raw = initial_structural_level
        current_structural_augmented = current_structural_raw
        nodes_per_chunk = tf.cast(nodes_per_chunk, tf.int32)
        default_nodes_per_chunk = tf.constant(-1, dtype=tf.int32)
        current_nodes_per_chunk = tf.where(
            nodes_per_chunk > 0, nodes_per_chunk, default_nodes_per_chunk
        )
        if training and self.memory_augmentor is not None:
            synthetic_level, augment_metrics = self.memory_augmentor(
                current_level_raw, training=training, return_losses=True
            )
            current_level_augmented = tf.concat([current_level_raw, synthetic_level], axis=1)
            synthetic_structural = synthetic_level
            current_structural_augmented = tf.concat(
                [current_structural_raw, synthetic_structural], axis=1
            )

            # Extract and accumulate QGAN losses if present
            if "qgan_total_loss" in augment_metrics:
                qgan_losses_ta = qgan_losses_ta.write(
                    qgan_loss_idx, augment_metrics["qgan_total_loss"]
                )
                qgan_loss_idx = qgan_loss_idx + 1

            if return_aux_metrics:
                level_str = tf.strings.as_string(tf.constant(0, dtype=tf.int32))
                prefix = tf.strings.join(
                    [tf.constant("memory_level_"), level_str, tf.constant("/qgan/")]
                )
                for key, value in augment_metrics.items():
                    # Skip loss keys from metrics - they're handled separately
                    # Use Python string operations (not TensorFlow) for graph mode compatibility
                    if not key.endswith("_loss"):
                        metric_keys_ta = metric_keys_ta.write(
                            metric_idx, tf.strings.join([prefix, tf.constant(key)])
                        )
                        metric_values_ta = metric_values_ta.write(metric_idx, value)
                        metric_idx = metric_idx + 1
        current_level_augmented.set_shape([None, None, self.compressed_dim])
        current_level_augmented.set_shape([None, None, self.compressed_dim])
        current_structural_augmented.set_shape([None, None, self.compressed_dim])
        ta = ta.write(0, current_level_augmented)

        i = tf.constant(1)

        def cond_mem(
            i,
            current_level_raw,
            current_level_augmented,
            current_structural_raw,
            current_structural_augmented,
            current_nodes_per_chunk,
            ta,
            metric_keys_ta,
            metric_values_ta,
            metric_idx,
            qgan_losses_ta,
            qgan_loss_idx,
        ):
            return tf.logical_and(
                tf.shape(current_level_raw)[1] > 1, i < config.MAX_HIERARCHY_LEVELS
            )

        def body_mem(
            i,
            current_level_raw,
            current_level_augmented,
            current_structural_raw,
            current_structural_augmented,
            current_nodes_per_chunk,
            ta,
            metric_keys_ta,
            metric_values_ta,
            metric_idx,
            qgan_losses_ta,
            qgan_loss_idx,
        ):
            """Loop body: performs one level of graph-based memory aggregation."""
            if config.DEBUG_MODE:
                tf.print(
                    "[HSMN while_loop] Iteration:",
                    i,
                    "Input shape:",
                    tf.shape(current_level_augmented),
                    output_stream=sys.stderr,
                )

            if return_aux_metrics:
                # ENTERPRISE FIX: Handle dimension mismatches with projection layers
                if hasattr(self, "dimension_mismatch") and self.dimension_mismatch:
                    # Project to graph_learner's expected dimensions
                    projected_level = self.project_to_graph(current_level_augmented)
                    projected_structural = self.project_to_graph(current_structural_augmented)

                    next_level_raw, next_structural_raw, aux_metrics = self.graph_learner(
                        projected_level,
                        structural_features=projected_structural,
                        nodes_per_chunk=current_nodes_per_chunk,
                        level_index=i,
                        training=training,
                        return_aux_metrics=True,
                    )

                    # Project back to compressed dimensions
                    next_level_raw = self.project_from_graph(next_level_raw)
                    next_structural_raw = self.project_from_graph(next_structural_raw)
                else:
                    # No dimension mismatch - call directly
                    next_level_raw, next_structural_raw, aux_metrics = self.graph_learner(
                        current_level_augmented,
                        structural_features=current_structural_augmented,
                        nodes_per_chunk=current_nodes_per_chunk,
                        level_index=i,
                        training=training,
                        return_aux_metrics=True,
                    )
                for key, value in aux_metrics.items():
                    full_key = tf.strings.join(
                        ["graph_learner_level_", tf.strings.as_string(i), "/", key]
                    )
                    metric_keys_ta = metric_keys_ta.write(metric_idx, full_key)

                    # Reduce batched metrics to scalars (mean over batch dimension)
                    # GAQuantumGraphLearner may return batched metrics with shape [batch_size]
                    # TensorArray expects scalar values for consistent shape across iterations
                    if len(value.shape) > 0:
                        scalar_value = tf.reduce_mean(value)
                    else:
                        scalar_value = value

                    metric_values_ta = metric_values_ta.write(metric_idx, scalar_value)
                    metric_idx = metric_idx + 1
            else:
                # ENTERPRISE FIX: Handle dimension mismatches with projection layers
                if hasattr(self, "dimension_mismatch") and self.dimension_mismatch:
                    # Project to graph_learner's expected dimensions
                    projected_level = self.project_to_graph(current_level_augmented)
                    projected_structural = self.project_to_graph(current_structural_augmented)

                    next_level_raw, next_structural_raw = self.graph_learner(
                        projected_level,
                        structural_features=projected_structural,
                        nodes_per_chunk=current_nodes_per_chunk,
                        level_index=i,
                        training=training,
                        return_aux_metrics=False,
                    )

                    # Project back to compressed dimensions
                    next_level_raw = self.project_from_graph(next_level_raw)
                    next_structural_raw = self.project_from_graph(next_structural_raw)
                else:
                    # No dimension mismatch - call directly
                    next_level_raw, next_structural_raw = self.graph_learner(
                        current_level_augmented,
                        structural_features=current_structural_augmented,
                        nodes_per_chunk=current_nodes_per_chunk,
                        level_index=i,
                        training=training,
                        return_aux_metrics=False,
                    )

            next_level_raw.set_shape([None, None, self.compressed_dim])
            next_structural_raw.set_shape([None, None, self.compressed_dim])

            augmented_next_level = next_level_raw
            augmented_structural = next_structural_raw
            if training and self.memory_augmentor is not None:
                synthetic_next, augment_metrics = self.memory_augmentor(
                    next_level_raw, training=training, return_losses=True
                )
                augmented_next_level = tf.concat([next_level_raw, synthetic_next], axis=1)
                augmented_next_level.set_shape([None, None, self.compressed_dim])
                synthetic_structural_next = synthetic_next
                augmented_structural = tf.concat(
                    [next_structural_raw, synthetic_structural_next], axis=1
                )
                augmented_structural.set_shape([None, None, self.compressed_dim])

                # Extract and accumulate QGAN losses if present
                if "qgan_total_loss" in augment_metrics:
                    qgan_losses_ta = qgan_losses_ta.write(
                        qgan_loss_idx, augment_metrics["qgan_total_loss"]
                    )
                    qgan_loss_idx = qgan_loss_idx + 1

                if return_aux_metrics:
                    full_prefix = tf.strings.join(
                        [
                            tf.constant("memory_level_"),
                            tf.strings.as_string(i),
                            tf.constant("/qgan/"),
                        ]
                    )
                    for key, value in augment_metrics.items():
                        # Skip loss keys from metrics - they're handled separately
                        # Use Python string operations (not TensorFlow) for graph mode compatibility
                        if not key.endswith("_loss"):
                            metric_keys_ta = metric_keys_ta.write(
                                metric_idx, tf.strings.join([full_prefix, tf.constant(key)])
                            )
                            metric_values_ta = metric_values_ta.write(metric_idx, value)
                            metric_idx = metric_idx + 1
            else:
                augmented_next_level.set_shape([None, None, self.compressed_dim])
                augmented_structural.set_shape([None, None, self.compressed_dim])

            ta = ta.write(i, augmented_next_level)

            if config.DEBUG_MODE:
                tf.print(
                    "[HSMN while_loop] Iteration:",
                    i,
                    "Output shape:",
                    tf.shape(augmented_next_level),
                    output_stream=sys.stderr,
                )

            return (
                i + 1,
                next_level_raw,
                augmented_next_level,
                next_structural_raw,
                augmented_structural,
                default_nodes_per_chunk,
                ta,
                metric_keys_ta,
                metric_values_ta,
                metric_idx,
                qgan_losses_ta,
                qgan_loss_idx,
            )

        (
            final_i,
            _,
            _,
            _,
            _,
            _,
            memory_levels_ta,
            final_metric_keys_ta,
            final_metric_values_ta,
            _,
            final_qgan_losses_ta,
            _,
        ) = tf.while_loop(
            cond_mem,
            body_mem,
            loop_vars=[
                i,
                current_level_raw,
                current_level_augmented,
                current_structural_raw,
                current_structural_augmented,
                current_nodes_per_chunk,
                ta,
                metric_keys_ta,
                metric_values_ta,
                metric_idx,
                qgan_losses_ta,
                qgan_loss_idx,
            ],
            shape_invariants=[
                i.get_shape(),
                tf.TensorShape([None, None, self.compressed_dim]),
                tf.TensorShape([None, None, self.compressed_dim]),
                tf.TensorShape([None, None, self.compressed_dim]),
                tf.TensorShape([None, None, self.compressed_dim]),
                tf.TensorShape(None),  # Allow shape to vary (handles both (None,) and () shapes)
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                metric_idx.get_shape(),
                tf.TensorShape([]),
                qgan_loss_idx.get_shape(),
            ],
        )
        # --- START: DEFINITIVE FIX ---
        # The loop counter `final_i` from tf.while_loop defaults to float32.
        # The subsequent `_stack_memory_levels` loop uses this as a boundary condition,
        # and its C++ implementation expects an integer, causing the fatal error.
        # We must cast it to int32 here to ensure type safety.
        num_memory_levels = tf.cast(final_i, dtype=tf.int32)
        if config.DEBUG_MODE:
            tf.print(
                "[MemoryHierarchyBuilder] Built hierarchy. Num levels:",
                num_memory_levels,
                "dtype:",
                num_memory_levels.dtype,
                output_stream=sys.stderr,
            )

        # Stack the collected keys and values into single tensors to be returned.
        final_metric_keys = final_metric_keys_ta.stack()
        final_metric_values = final_metric_values_ta.stack()

        # Accumulate and add QGAN losses from the while loop
        # These losses were collected inside the loop and must be added here to avoid scope issues
        if training and self.memory_augmentor is not None:
            stacked_qgan_losses = final_qgan_losses_ta.stack()
            # Sum all losses (if empty, sum will be 0 which is harmless)
            # We avoid using tf.size() in a Python if statement for graph mode compatibility
            total_qgan_loss = tf.reduce_sum(stacked_qgan_losses)
            self.add_loss(total_qgan_loss)
            if config.DEBUG_MODE:
                tf.print(
                    "[MemoryHierarchyBuilder] Added accumulated QGAN loss:",
                    total_qgan_loss,
                    output_stream=sys.stderr,
                )

        return memory_levels_ta, num_memory_levels, (final_metric_keys, final_metric_values)
        # --- END: DEFINITIVE FIX ---

    def _stack_memory_levels(
        self, memory_levels_ta: tf.TensorArray, num_memory_levels: tf.Tensor, batch_size: int
    ) -> tf.Tensor:
        """Decompresses and stacks all memory levels into a single sequence."""
        i_stack = tf.constant(0)
        initial_memory = tf.zeros(
            [batch_size, 0, self.embedding_dim], dtype=tf.float32, name="initial_stacked_memory"
        )

        def cond_stack(i, stacked_memory, ta_in):
            return i < num_memory_levels

        def body_stack(i, stacked_memory, ta_in):
            level_embeddings = ta_in.read(i)
            level_embeddings.set_shape([None, None, self.compressed_dim])
            shape = tf.shape(level_embeddings)
            batch_size_body, num_nodes_body = shape[0], shape[1]
            reshaped_for_dense = tf.reshape(level_embeddings, [-1, self.compressed_dim])
            decompressed_flat = self.decompressor(reshaped_for_dense)
            decompressed_level = tf.reshape(
                decompressed_flat, [batch_size_body, num_nodes_body, self.embedding_dim]
            )
            lvl_emb = self.level_embeddings(i)
            lvl_emb_expanded = tf.reshape(lvl_emb, [1, 1, self.embedding_dim])
            decompressed_level_with_pos = decompressed_level + lvl_emb_expanded
            stacked_memory = tf.concat([stacked_memory, decompressed_level_with_pos], axis=1)
            if config.DEBUG_MODE:
                tf.print(
                    "[MemoryHierarchyBuilder._stack_memory_levels] Loop:",
                    i,
                    "Num Levels:",
                    num_memory_levels,
                    "i.dtype:",
                    i.dtype,
                    "num_levels.dtype:",
                    num_memory_levels.dtype,
                    output_stream=sys.stderr,
                )

            return i + 1, stacked_memory, ta_in

        _, stacked_memory, _ = tf.while_loop(
            cond_stack,
            body_stack,
            loop_vars=[i_stack, initial_memory, memory_levels_ta],
            shape_invariants=[
                i_stack.get_shape(),
                tf.TensorShape([None, None, self.embedding_dim]),
                tf.TensorShape([]),
            ],
        )
        return stacked_memory

    def call(
        self,
        initial_memory_level: tf.Tensor,
        initial_structural_level: tf.Tensor,
        nodes_per_chunk: tf.Tensor,
        training: bool = False,
        return_aux_metrics: bool = False,
    ) -> tuple[tf.Tensor, tf.TensorArray, tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """
        Main call method to build and stack the memory hierarchy.

        Args:
            initial_memory_level: The level 0 memory tensor.
            training: Boolean flag for training mode.
            return_aux_metrics: If True, returns auxiliary metrics from sub-layers.

        Returns:
            A tuple of (stacked_memory, memory_levels_ta, num_memory_levels, aux_metrics_tensors).
            The aux_metrics_tensors is a tuple of (keys, values).
        """
        batch_size = tf.shape(initial_memory_level)[0]
        memory_levels_ta, num_memory_levels, aux_metrics_tensors = self._build_memory_hierarchy(
            initial_memory_level,
            initial_structural_level,
            nodes_per_chunk,
            training,
            return_aux_metrics=return_aux_metrics,
        )
        stacked_memory = self._stack_memory_levels(memory_levels_ta, num_memory_levels, batch_size)
        return stacked_memory, memory_levels_ta, num_memory_levels, aux_metrics_tensors
