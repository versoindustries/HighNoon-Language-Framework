# highnoon/_native/ops/segment_softmax.py
# Copyright 2025 Verso Industries
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

"""Segment softmax utility for MoE routing."""

import tensorflow as tf


def segment_softmax(
    data: tf.Tensor, segment_ids: tf.Tensor, num_segments: int | None = None
) -> tf.Tensor:
    """
    Computes a softmax over segments of a tensor.

    This function is a custom implementation equivalent to `tf.math.segment_softmax`,
    built using more fundamental TensorFlow operations to ensure compatibility across
    different versions. It computes the softmax independently for each segment of the data.

    Args:
        data: A 1-D tensor. The data to be softmaxed (e.g., logits).
        segment_ids: A 1-D tensor with the same size as `data`. The IDs for the
                     segmentation. Values should be sorted for performance, but it's
                     not a strict requirement.
        num_segments: An optional integer specifying the number of distinct segment IDs.
                      If not provided, it's inferred from `segment_ids`, which can be
                      less efficient.

    Returns:
        A tensor with the same shape as `data` with softmax applied to each segment.
    """
    if num_segments is None:
        num_segments = tf.reduce_max(segment_ids) + 1

    # For numerical stability, we subtract the maximum value from each segment
    # before exponentiating. This prevents overflow for large logits.
    max_per_segment = tf.math.segment_max(data, segment_ids)
    # Use tf.gather to broadcast the max value of each segment to all its elements.
    data_max_subtracted = data - tf.gather(max_per_segment, segment_ids)

    # Exponentiate the stabilized data.
    exp_data = tf.exp(data_max_subtracted)

    # Sum the exponentiated values for each segment to get the softmax denominator.
    sum_per_segment = tf.math.segment_sum(exp_data, segment_ids)
    # Use tf.gather again to broadcast the denominator to all elements in a segment.
    denominator = tf.gather(sum_per_segment, segment_ids)

    # To avoid division by zero for empty segments (which would result in NaN),
    # we replace any zero denominators with ones. The corresponding numerators
    # for these segments will also be zero, so the output will be correct (0).
    safe_denominator = tf.where(denominator > 0, denominator, tf.ones_like(denominator))

    softmax_output = exp_data / safe_denominator

    return softmax_output
