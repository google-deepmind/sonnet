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

"""Relational Memory architecture.

An implementation of the architecture described in "Relational Recurrent
Neural Networks", Santoro et al., 2018.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from sonnet.python.modules import basic
from sonnet.python.modules import layer_norm
from sonnet.python.modules import rnn_core
from sonnet.python.modules.nets import mlp
import tensorflow as tf


class RelationalMemory(rnn_core.RNNCore):
  """Relational Memory Core."""

  def __init__(self, mem_slots, head_size, num_heads=1, num_blocks=1,
               forget_bias=1.0, input_bias=0.0, gate_style='unit',
               attention_mlp_layers=2, key_size=None, name='relational_memory'):
    """Constructs a `RelationalMemory` object.

    Args:
      mem_slots: The total number of memory slots to use.
      head_size: The size of an attention head.
      num_heads: The number of attention heads to use. Defaults to 1.
      num_blocks: Number of times to compute attention per time step. Defaults
        to 1.
      forget_bias: Bias to use for the forget gate, assuming we are using
        some form of gating. Defaults to 1.
      input_bias: Bias to use for the input gate, assuming we are using
        some form of gating. Defaults to 0.
      gate_style: Whether to use per-element gating ('unit'),
        per-memory slot gating ('memory'), or no gating at all (None).
        Defaults to `unit`.
      attention_mlp_layers: Number of layers to use in the post-attention
        MLP. Defaults to 2.
      key_size: Size of vector to use for key & query vectors in the attention
        computation. Defaults to None, in which case we use `head_size`.
      name: Name of the module.

    Raises:
      ValueError: gate_style not one of [None, 'memory', 'unit'].
      ValueError: num_blocks is < 1.
      ValueError: attention_mlp_layers is < 1.
    """
    super(RelationalMemory, self).__init__(name=name)

    self._mem_slots = mem_slots
    self._head_size = head_size
    self._num_heads = num_heads
    self._mem_size = self._head_size * self._num_heads

    if num_blocks < 1:
      raise ValueError('num_blocks must be >= 1. Got: {}.'.format(num_blocks))
    self._num_blocks = num_blocks

    self._forget_bias = forget_bias
    self._input_bias = input_bias

    if gate_style not in ['unit', 'memory', None]:
      raise ValueError(
          'gate_style must be one of [\'unit\', \'memory\', None]. Got: '
          '{}.'.format(gate_style))
    self._gate_style = gate_style

    if attention_mlp_layers < 1:
      raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.format(
          attention_mlp_layers))
    self._attention_mlp_layers = attention_mlp_layers

    self._key_size = key_size if key_size else self._head_size

  def initial_state(self, batch_size, trainable=False):
    """Creates the initial memory.

    We should ensure each row of the memory is initialized to be unique,
    so initialize the matrix to be the identity. We then pad or truncate
    as necessary so that init_state is of size
    (batch_size, self._mem_slots, self._mem_size).

    Args:
      batch_size: The size of the batch.
      trainable: Whether the initial state is trainable. This is always True.

    Returns:
      init_state: A truncated or padded matrix of size
        (batch_size, self._mem_slots, self._mem_size).
    """
    init_state = tf.eye(self._mem_slots, batch_shape=[batch_size])

    # Pad the matrix with zeros.
    if self._mem_size > self._mem_slots:
      difference = self._mem_size - self._mem_slots
      pad = tf.zeros((batch_size, self._mem_slots, difference))
      init_state = tf.concat([init_state, pad], -1)
    # Truncation. Take the first `self._mem_size` components.
    elif self._mem_size < self._mem_slots:
      init_state = init_state[:, :, :self._mem_size]
    return init_state

  def _multihead_attention(self, memory):
    """Perform multi-head attention from 'Attention is All You Need'.

    Implementation of the attention mechanism from
    https://arxiv.org/abs/1706.03762.

    Args:
      memory: Memory tensor to perform attention on.

    Returns:
      new_memory: New memory tensor.
    """
    key_size = self._key_size
    value_size = self._head_size

    qkv_size = 2 * key_size + value_size
    total_size = qkv_size * self._num_heads  # Denote as F.
    qkv = basic.BatchApply(basic.Linear(total_size))(memory)
    qkv = basic.BatchApply(layer_norm.LayerNorm())(qkv)

    mem_slots = memory.get_shape().as_list()[1]  # Denoted as N.

    # [B, N, F] -> [B, N, H, F/H]
    qkv_reshape = basic.BatchReshape([mem_slots, self._num_heads,
                                      qkv_size])(qkv)

    # [B, N, H, F/H] -> [B, H, N, F/H]
    qkv_transpose = tf.transpose(qkv_reshape, [0, 2, 1, 3])
    q, k, v = tf.split(qkv_transpose, [key_size, key_size, value_size], -1)

    q *= qkv_size ** -0.5
    dot_product = tf.matmul(q, k, transpose_b=True)  # [B, H, N, N]
    weights = tf.nn.softmax(dot_product)

    output = tf.matmul(weights, v)  # [B, H, N, V]

    # [B, H, N, V] -> [B, N, H, V]
    output_transpose = tf.transpose(output, [0, 2, 1, 3])

    # [B, N, H, V] -> [B, N, H * V]
    new_memory = basic.BatchFlatten(preserve_dims=2)(output_transpose)
    return new_memory

  @property
  def state_size(self):
    return tf.TensorShape([self._mem_slots, self._mem_size])

  @property
  def output_size(self):
    return tf.TensorShape(self._mem_slots * self._mem_size)

  def _calculate_gate_size(self):
    """Calculate the gate size from the gate_style.

    Returns:
      The per sample, per head parameter size of each gate.
    """
    if self._gate_style == 'unit':
      return self._mem_size
    elif self._gate_style == 'memory':
      return 1
    else:  # self._gate_style == None
      return 0

  def _create_gates(self, inputs, memory):
    """Create input and forget gates for this step using `inputs` and `memory`.

    Args:
      inputs: Tensor input.
      memory: The current state of memory.

    Returns:
      input_gate: A LSTM-like insert gate.
      forget_gate: A LSTM-like forget gate.
    """
    # We'll create the input and forget gates at once. Hence, calculate double
    # the gate size.
    num_gates = 2 * self._calculate_gate_size()

    memory = tf.tanh(memory)
    inputs = basic.BatchFlatten()(inputs)
    gate_inputs = basic.BatchApply(basic.Linear(num_gates), n_dims=1)(inputs)
    gate_inputs = tf.expand_dims(gate_inputs, axis=1)
    gate_memory = basic.BatchApply(basic.Linear(num_gates))(memory)
    gates = tf.split(gate_memory + gate_inputs, num_or_size_splits=2, axis=2)
    input_gate, forget_gate = gates

    input_gate = tf.sigmoid(input_gate + self._input_bias)
    forget_gate = tf.sigmoid(forget_gate + self._forget_bias)

    return input_gate, forget_gate

  def _attend_over_memory(self, memory):
    """Perform multiheaded attention over `memory`.

    Args:
      memory: Current relational memory.

    Returns:
      The attended-over memory.
    """
    attention_mlp = basic.BatchApply(
        mlp.MLP([self._mem_size] * self._attention_mlp_layers))
    for _ in range(self._num_blocks):
      attended_memory = self._multihead_attention(memory)

      # Add a skip connection to the multiheaded attention's input.
      memory = basic.BatchApply(layer_norm.LayerNorm())(
          memory + attended_memory)

      # Add a skip connection to the attention_mlp's input.
      memory = basic.BatchApply(layer_norm.LayerNorm())(
          attention_mlp(memory) + memory)

    return memory

  def _build(self, inputs, memory, treat_input_as_matrix=False):
    """Adds relational memory to the TensorFlow graph.

    Args:
      inputs: Tensor input.
      memory: Memory output from the previous time step.
      treat_input_as_matrix: Optional, whether to treat `input` as a sequence
        of matrices. Defaulta to False, in which case the input is flattened
        into a vector.

    Returns:
      output: This time step's output.
      next_memory: The next version of memory to use.
    """
    if treat_input_as_matrix:
      inputs = basic.BatchFlatten(preserve_dims=2)(inputs)
      inputs_reshape = basic.BatchApply(
          basic.Linear(self._mem_size), n_dims=2)(inputs)
    else:
      inputs = basic.BatchFlatten()(inputs)
      inputs = basic.Linear(self._mem_size)(inputs)
      inputs_reshape = tf.expand_dims(inputs, 1)

    memory_plus_input = tf.concat([memory, inputs_reshape], axis=1)
    next_memory = self._attend_over_memory(memory_plus_input)

    n = inputs_reshape.get_shape().as_list()[1]
    next_memory = next_memory[:, :-n, :]

    if self._gate_style == 'unit' or self._gate_style == 'memory':
      self._input_gate, self._forget_gate = self._create_gates(
          inputs_reshape, memory)
      next_memory = self._input_gate * tf.tanh(next_memory)
      next_memory += self._forget_gate * memory

    output = basic.BatchFlatten()(next_memory)
    return output, next_memory

  @property
  def input_gate(self):
    """Returns the input gate Tensor."""
    self._ensure_is_connected()
    return self._input_gate

  @property
  def forget_gate(self):
    """Returns the forget gate Tensor."""
    self._ensure_is_connected()
    return self._forget_gate
