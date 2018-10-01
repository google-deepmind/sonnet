# Copyright 2017 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Modules for attending over memory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

import numpy as np
from sonnet.python.modules import base
from sonnet.python.modules import basic
import tensorflow as tf


# Result of AttentiveRead._build(). See docstring therein for details.
AttentionOutput = collections.namedtuple(
    "AttentionOutput", ["read", "weights", "weight_logits"])


class AttentiveRead(base.AbstractModule):
  """A module for reading with attention.

  This module reads a weighted sum of embeddings from memory, where each
  memory slot's weight is based on the logit returned by an attention embedding
  module. A mask may be given to ignore some memory slots (e.g. when attending
  over variable-length sequences).
  """

  def __init__(self, attention_logit_mod, name="attention"):
    """Initialize AttentiveRead module.

    Args:
      attention_logit_mod: Module that produces logit corresponding to a memory
        slot's compatibility. Must map a [batch_size * memory_size,
        memory_word_size + query_word_size]-shaped Tensor to a
        [batch_size * memory_size, 1] shape Tensor.
      name: string. Name for module.
    """
    super(AttentiveRead, self).__init__(name=name)

    self._attention_logit_mod = attention_logit_mod

  def _build(self, memory, query, memory_mask=None):
    """Perform a differentiable read.

    Args:
      memory: [batch_size, memory_size, memory_word_size]-shaped Tensor of
        dtype float32. This represents, for each example and memory slot, a
        single embedding to attend over.
      query: [batch_size, query_word_size]-shaped Tensor of dtype float32.
        Represents, for each example, a single embedding representing a query.
      memory_mask: None or [batch_size, memory_size]-shaped Tensor of dtype
        bool. An entry of False indicates that a memory slot should not enter
        the resulting weighted sum. If None, all memory is used.

    Returns:
      An AttentionOutput instance containing:
        read: [batch_size, memory_word_size]-shaped Tensor of dtype float32.
          This represents, for each example, a weighted sum of the contents of
          the memory.
        weights: [batch_size, memory_size]-shaped Tensor of dtype float32. This
          represents, for each example and memory slot, the attention weights
          used to compute the read.
        weight_logits: [batch_size, memory_size]-shaped Tensor of dtype float32.
          This represents, for each example and memory slot, the logits of the
          attention weights, that is, `weights` is calculated by taking the
          softmax of the weight logits.

    Raises:
      UnderspecifiedError: if memory_word_size or query_word_size can not be
        inferred.
      IncompatibleShapeError: if memory, query, memory_mask, or output of
        attention_logit_mod do not match expected shapes.
    """
    if len(memory.get_shape()) != 3:
      raise base.IncompatibleShapeError(
          "memory must have shape [batch_size, memory_size, memory_word_size].")

    if len(query.get_shape()) != 2:
      raise base.IncompatibleShapeError(
          "query must have shape [batch_size, query_word_size].")

    if memory_mask is not None and len(memory_mask.get_shape()) != 2:
      raise base.IncompatibleShapeError(
          "memory_mask must have shape [batch_size, memory_size].")

    # Ensure final dimensions are defined, else the attention logit module will
    # be unable to infer input size when constructing variables.
    inferred_memory_word_size = memory.get_shape()[2].value
    inferred_query_word_size = query.get_shape()[1].value
    if inferred_memory_word_size is None or inferred_query_word_size is None:
      raise base.UnderspecifiedError(
          "memory_word_size and query_word_size must be known at graph "
          "construction time.")

    memory_shape = tf.shape(memory)
    batch_size = memory_shape[0]
    memory_size = memory_shape[1]

    query_shape = tf.shape(query)
    query_batch_size = query_shape[0]

    # Transform query to have same number of words as memory.
    #
    # expanded_query: [batch_size, memory_size, query_word_size].
    expanded_query = tf.tile(tf.expand_dims(query, dim=1), [1, memory_size, 1])

    # Compute attention weights for each memory slot.
    #
    # attention_weight_logits: [batch_size, memory_size]
    with tf.control_dependencies(
        [tf.assert_equal(batch_size, query_batch_size)]):
      concatenated_embeddings = tf.concat(
          values=[memory, expanded_query], axis=2)

    batch_apply_attention_logit = basic.BatchApply(
        self._attention_logit_mod, n_dims=2, name="batch_apply_attention_logit")
    attention_weight_logits = batch_apply_attention_logit(
        concatenated_embeddings)

    # Note: basic.BatchApply() will automatically reshape the [batch_size *
    # memory_size, 1]-shaped result of self._attention_logit_mod(...) into a
    # [batch_size, memory_size, 1]-shaped Tensor. If
    # self._attention_logit_mod(...) returns something with more dimensions,
    # then attention_weight_logits will have extra dimensions, too.
    if len(attention_weight_logits.get_shape()) != 3:
      raise base.IncompatibleShapeError(
          "attention_weight_logits must be a rank-3 Tensor. Are you sure that "
          "attention_logit_mod() returned [batch_size * memory_size, 1]-shaped"
          " Tensor?")

    # Remove final length-1 dimension.
    attention_weight_logits = tf.squeeze(attention_weight_logits, [2])

    # Mask out ignored memory slots by assigning them very small logits. Ensures
    # that every example has at least one valid memory slot, else we'd end up
    # averaging all memory slots equally.
    if memory_mask is not None:
      num_remaining_memory_slots = tf.reduce_sum(
          tf.cast(memory_mask, dtype=tf.int32), axis=[1])
      with tf.control_dependencies(
          [tf.assert_positive(num_remaining_memory_slots)]):
        finfo = np.finfo(np.float32)
        kept_indices = tf.cast(memory_mask, dtype=tf.float32)
        ignored_indices = tf.cast(tf.logical_not(memory_mask), dtype=tf.float32)
        lower_bound = finfo.max * kept_indices + finfo.min * ignored_indices
        attention_weight_logits = tf.minimum(attention_weight_logits,
                                             lower_bound)

    # attended_memory: [batch_size, memory_word_size].
    attention_weight = tf.reshape(
        tf.nn.softmax(attention_weight_logits),
        shape=[batch_size, memory_size, 1])
    # The multiplication is elementwise and relies on broadcasting the weights
    # across memory_word_size. Then we sum across the memory slots.
    attended_memory = tf.reduce_sum(memory * attention_weight, axis=[1])

    # Infer shape of result as much as possible.
    inferred_batch_size, _, inferred_memory_word_size = (
        memory.get_shape().as_list())
    attended_memory.set_shape([inferred_batch_size, inferred_memory_word_size])

    return AttentionOutput(
        read=attended_memory,
        weights=tf.squeeze(attention_weight, [2]),
        weight_logits=attention_weight_logits)
