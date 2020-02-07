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
"""Implementation of Transformer networks.

Size glossary:
  * Batch size (B).
  * Sequence length (N).
  * Memory size (M). The size of the optional memory, passed in via `state`.
  * Number of heads (H): the number of attention heads.
  * Value size (V): the size of each value embedding per head.
  * Key size (K): the size of each key embedding per head. Equally, the size
      of each query embedding per head. Typically K <= V.
  * Embedding size (HV). The size of the activation or embedding relating to
      each input between layers. Equal to value_size * num_heads.
  * All attention size (F). The size of all attention activations over every
      head.
  * QKV size (F / H): The size of the query, key and value per head. Equal to
      2K + V or equivalently F / H.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import numpy as np
from sonnet.python import custom_getters
from sonnet.python.modules import base
from sonnet.python.modules import basic
from sonnet.python.modules import conv as snt_conv
from sonnet.python.modules import layer_norm as snt_ln
from sonnet.python.modules import rnn_core
from sonnet.python.modules import util
from sonnet.python.modules.nets import mlp as snt_mlp
import tensorflow.compat.v1 as tf

AttentionState = collections.namedtuple('AttentionState',
                                        ('queries', 'keys', 'values', 'logits',
                                         'weights', 'embeddings', 'read_words'))

CompressedMemoryState = collections.namedtuple(
    'CompressedMemoryState', ('episodic_memory', 'compressed_memory', 'index'))


def rel_shift(position_logits):
  """Shifting of logits for relative attention.

  Args:
    position_logits: A tensor of shape [B, H, N, N + M].

  Returns:
    The shifted logits. Example, for input (H=1, B=1):
      [5, 4, 3, 2, 1]
      [5, 4, 3, 2, 1]
      [5, 4, 3, 2, 1]
      [5, 4, 3, 2, 1]
      [5, 4, 3, 2, 1]

    the function outputs:
      [1, 0, 5, 4, 3]
      [2, 1, 0, 5, 4]
      [3, 2, 1, 0, 5]
      [4, 3, 2, 1, 0]
      [5, 4, 3, 2, 1]

  Raises:
    ValueError if position_logits is not 4D.

  Note: this is not an exact shift as the upper triangle is non-zero. This
  works as intended in the causally-masked case. If this is used with un-masked
  attention, we'd want these to also be zero.
  """
  if position_logits.get_shape().ndims != 4:
    raise ValueError('Expected 4D position logits.')

  input_shape = tf.shape(position_logits)
  batch_size = input_shape[0]
  num_heads = input_shape[1]
  t1 = input_shape[2]
  t2 = input_shape[3]
  # We prepend zeros on the final timescale dimension.
  to_pad = tf.zeros([batch_size, num_heads, t1, 1])
  position_logits = tf.concat([to_pad, position_logits], -1)
  # Reshape trick to shift input.
  position_logits = tf.reshape(position_logits,
                               [batch_size, num_heads, t2 + 1, t1])
  # Remove extra time dimension and re-shape.
  position_logits = position_logits[:, :, 1:]
  position_logits = tf.reshape(position_logits, input_shape)
  return position_logits


def _layer_norm(inputs):
  if inputs.get_shape().ndims > 2:
    return basic.BatchApply(snt_ln.LayerNorm())(inputs)
  else:
    return snt_ln.LayerNorm()(inputs)


def _concat_and_slice(prev_memory, new_memory):
  original_memory_size = prev_memory.get_shape().as_list()[1]
  concat_memory = tf.concat([prev_memory, new_memory], 1)
  memory = concat_memory[:, -original_memory_size:]
  return memory, concat_memory


def simple_attention(queries, keys, values):
  logits = tf.matmul(queries, keys, transpose_b=True)
  weights = tf.nn.softmax(logits)
  return tf.matmul(weights, values)


class ResidualDropoutWrapper(base.AbstractModule):
  """Wrapper class that applies residual connections, dropout and layer norm.

  By default applies a relu to the module output before the other operations.
  """

  def __init__(self,
               layer,
               dropout_rate,
               layer_norm='input',
               name='residual_dropout_wrapper'):
    self._module = layer
    self._dropout_rate = dropout_rate
    self._layer_norm = layer_norm
    super(ResidualDropoutWrapper, self).__init__(name=name)

  def _build(self, inputs, *args, **kwargs):
    if self._layer_norm in ('both', 'input'):
      normed_inputs = _layer_norm(inputs)
    else:
      normed_inputs = inputs
    module_output = self._module(normed_inputs, *args, **kwargs)
    module_state = None
    # If module outputs multiple items, assumes (output, state) tuple.
    if isinstance(module_output, tuple):
      module_output, module_state = module_output
    if kwargs['is_training']:  # kwargs must contain is_training.
      module_output = tf.nn.dropout(module_output, rate=self._dropout_rate)
    output = inputs + module_output
    if self._layer_norm in ('both', 'output'):
      output = _layer_norm(output)
    if module_state is None:
      return output
    else:
      return output, module_state


def future_mask(chunk_size, dtype):
  """Creates attention mask to ensure an element i cannot attend to j > i."""
  square = tf.ones([chunk_size, chunk_size], dtype=dtype)
  # Create upper diagonal matrix and remove diagonal entries (allow self-attn).
  mask = tf.matrix_band_part(square, 0, -1) - tf.matrix_band_part(square, 0, 0)
  # Multiply by -1e6 and expand to broadcast with [B, H, N, N] logits.
  mask = -1e6 * tf.reshape(mask, [1, 1, chunk_size, chunk_size])
  return mask


def _memory_size(state):
  if isinstance(state, CompressedMemoryState):
    return (state.episodic_memory.get_shape().as_list()[1] +
            state.compressed_memory.get_shape().as_list()[1])
  else:
    return state.get_shape().as_list()[1]


def create_mask(inputs, state, equal_window):
  """Creates mask for future sequence positions.

  Args:
    inputs: inputs tensor of shape [B, N, D]
    state: optional tensor of shape [B, M, D], CompressedMemoryState or a list
      where the ith entry corresponds to the ith layer's state.
    equal_window: if True, then each activation has an equally-sized attention
      window of length 'M'. This only makes sense if a state is given.

  Returns:
    Float tensor of shape [1, 1, N, N + M], to be summed with logits.
  """
  chunk_size = inputs.get_shape().as_list()[1]
  dtype = inputs.dtype
  mask = future_mask(chunk_size, dtype)
  if state is not None:
    if isinstance(state, (tuple, list)):
      largest_memory_layer = np.argmax([_memory_size(s) for s in state])
      state = state[largest_memory_layer]
    mem_size = _memory_size(state)
    mask = tf.concat(
        [tf.zeros([1, 1, chunk_size, mem_size], dtype=dtype), mask], 3)

  if equal_window:
    attn_mask = tf.ones([chunk_size, chunk_size], dtype=dtype)
    mask_dia = tf.cast(tf.matrix_band_part(attn_mask, 0, 0), dtype=dtype)
    mask_l = tf.cast(tf.matrix_band_part(attn_mask, -1, 0), dtype=dtype)
    start_mask = tf.reshape(mask_l - mask_dia,
                            [1, 1, chunk_size, chunk_size]) * -1e6
    mask = tf.concat(
        [mask[:, :, :, :chunk_size] + start_mask, mask[:, :, :, chunk_size:]],
        3)
  return mask


def default_mlp(hidden_sizes, activate_final=False, init_std=2., **kwargs):
  """Standard batch-applied MLP for transformer modules."""
  init = {'w': tf.variance_scaling_initializer(init_std, distribution='normal')}
  mlp = snt_mlp.MLP(
      hidden_sizes,
      activate_final=activate_final,
      use_dropout=True,
      initializers=init,
      **kwargs)
  return basic.BatchApply(mlp)


def get_position_encodings(sequence_length,
                           hidden_size,
                           clamp_value,
                           max_timescale=10000.,
                           min_timescale=2.0):
  """Creates sinusoidal encodings of shape [1, N + M, D]."""
  # NOTE: when not using relative position encodings, min_timescale must be 2.0
  # and hidden_size must be an even number. Otherwise, the dimensions do not
  # match.
  pos_seq = tf.range(sequence_length - 1, -1, -1.0)
  if clamp_value > 0:
    pos_seq = tf.minimum(pos_seq, clamp_value)
  freqs = tf.range(0, hidden_size, min_timescale)
  inv_freq = 1 / (max_timescale**(freqs / hidden_size))
  sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
  pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
  pos_emb = tf.expand_dims(pos_emb, 0)

  output_dim = pos_emb.get_shape().as_list()[-1]
  if output_dim != hidden_size:
    raise ValueError(
        'position embedding dimension ({}) does not match that of the input ({}).'
        .format(output_dim, hidden_size))
  return pos_emb


class MultiheadAttention(base.AbstractModule):
  """Implements multi-head attention with optional state context."""

  def __init__(self,
               value_size,
               key_size,
               num_heads,
               mask=None,
               scaling=True,
               positional_encodings=None,
               use_relative_positions=False,
               init_std=2.,
               name='multihead_attention'):
    """Creates a MultiheadAttention module.

    Args:
      value_size: V parameter. See size glossary in class docstring.
      key_size: K parameter. See size glossary in class docstring.
      num_heads: The number of independent queries per timestep.
      mask: Optional mask to attention logits. This can prevent attending to
        future positions or unused memory slots.
      scaling: Whether to scale the attention logits.
      positional_encodings: Either None (none given), or an iterable of
        `(key_positional_encodings, query_positional_encodings)` tuples, where
        the first encodings in the list indicate the oldest entries in memory
        and the final encodings indicate the newest entries in memory and the
        sequence.
      use_relative_positions: If True then relative positions are incorporated,
        vs absolute, into the attention logits. This is done exactly as
        described in the TransformerXL, Dai et al. 2019.
      init_std: scaling of standard deviation for weight matrices init.
      name: Name of module.
    """

    super(MultiheadAttention, self).__init__(name=name)
    self._value_size = value_size
    self._key_size = key_size
    self._sizes = {
        'value': self._value_size,
        'key': self._key_size,
        'query': self._key_size,
        'relative_keys': self._key_size,
        'relative_keys_0': self._key_size,
    }
    self._num_heads = num_heads
    self._mask = mask
    self._scaling = scaling
    self._positional_encodings = positional_encodings
    self._use_relative_positions = use_relative_positions
    self._init = {'w': tf.variance_scaling_initializer(init_std)}

  @util.reuse_variables
  def multihead_linear(self, inputs, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      hidden_size = self._sizes[name]
      input_size = inputs.shape[-1].value
      w = tf.get_variable(
          'linear/w',
          shape=[input_size, self._num_heads * hidden_size],
          initializer=self._init['w'])
      w = tf.reshape(w, [input_size, self._num_heads, hidden_size])
      out = tf.einsum('bij,jhk->bhik', inputs, w)
      return out

  def _build(self,
             inputs,
             query_inputs=None,
             state=None,
             is_training=False,
             dropout_keep_prob=0.5):
    embedding_size = self._value_size * self._num_heads

    q_inputs = inputs if query_inputs is None else query_inputs
    # Denoted by L. If query_inputs is None, L = N.
    _, query_size = q_inputs.get_shape().as_list()[:2]

    if state is not None:
      if isinstance(state, CompressedMemoryState):
        state_memory_list = [state.compressed_memory, state.episodic_memory]
      else:
        state_memory_list = [state]

      k_inputs = tf.concat(state_memory_list + [inputs], 1)
      v_inputs = k_inputs
    else:
      k_inputs = inputs
      v_inputs = inputs

    # Batch size denoted by B
    batch_size = tf.shape(inputs)[0]
    # Chunk_size denoted by N
    chunk_size = inputs.get_shape().as_list()[1]
    # Denoted by N + M
    att_size = k_inputs.get_shape().as_list()[1]

    if self._positional_encodings and not self._use_relative_positions:
      key_positions, query_positions = self._positional_encodings
      k_inputs += key_positions
      q_inputs += query_positions

    # [B, H, L, K]
    q = self.multihead_linear(q_inputs, 'query')
    # [B, H, N + M, K]
    k = self.multihead_linear(k_inputs, 'key')
    # [B, H, N + M, V]
    v = self.multihead_linear(v_inputs, 'value')

    # Scaling the dot-product
    if self._scaling:
      q *= self._key_size**-0.5

    # [B, H, L, N + M]
    if self._use_relative_positions:
      r_w_bias = tf.get_variable(
          'r_w_bias', [1, self._num_heads, 1, self._key_size],
          dtype=inputs.dtype)
      content_logits = tf.matmul(q + r_w_bias, k, transpose_b=True)
      all_relative_logits = []
      # Loop over multiple positional encodings, for the case of multiple
      # memory types.
      for i, positional_encodings in enumerate(self._positional_encodings):
        key_positions, query_positions = positional_encodings
        if key_positions.get_shape().as_list()[-1] != att_size:
          key_positions = key_positions[:, -att_size:]  # Crop to layer mem size
        is_final = i == len(self._positional_encodings) - 1
        suffix = '' if is_final else '_%d' % i
        relative_keys = self.multihead_linear(
            key_positions, name='relative_keys' + suffix)
        # [B, H, N, D]
        r_r_bias = tf.get_variable(
            'r_r_bias' + suffix, [1, self._num_heads, 1, self._key_size],
            dtype=inputs.dtype)
        relative_keys = tf.tile(relative_keys, [batch_size, 1, 1, 1])
        relative_logits = tf.matmul(
            q + r_r_bias, relative_keys, transpose_b=True)
        relative_logits = rel_shift(relative_logits)
        if not is_final:  # Include relative positions for input sequence.
          relative_logits = relative_logits[:, :, :, :-chunk_size]
        all_relative_logits.append(relative_logits)
      all_relative_logits = tf.concat(all_relative_logits, 3)
      logits = content_logits + all_relative_logits
    else:
      # [B, H, N, N + M]
      logits = tf.matmul(q, k, transpose_b=True)
      content_logits = logits

    if self._mask is not None:
      if self._mask.get_shape().as_list()[-1] != att_size:
        mask = self._mask[:, :, :, -att_size:]
      else:
        mask = self._mask
      logits += mask

    weights = tf.nn.softmax(logits)
    if is_training:
      weights = tf.nn.dropout(weights, dropout_keep_prob)
    # [B, L, H, V], where V is value_size
    output_transpose = tf.einsum('bhij,bhjk->bihk', weights, v)

    # [B, L, H, V] -> [B, L, HV]
    attended_inputs = basic.BatchReshape([query_size, embedding_size])(
        output_transpose)
    # Apply final mlp to mix information between heads.
    output = basic.BatchApply(basic.Linear(embedding_size))(attended_inputs)

    attention_state = AttentionState(
        queries=q,
        keys=k,
        values=v,
        weights=weights,
        logits=content_logits,
        embeddings=inputs,
        read_words=output)
    return output, attention_state


class TransformerTower(base.AbstractModule):
  """Transformer tower.

  Deep residual network using blocks of attention and MLPs, specified in
  Vaswani et al. 2017.
  """

  def __init__(self,
               value_size,
               num_heads,
               num_layers,
               causal=True,
               key_size=None,
               shared_attention=False,
               output_size=None,
               mlp_hidden_sizes=tuple([1024]),
               dropout_rate=0.1,
               use_relative_positions=True,
               clamp_time_range=0,
               same_attention_length=False,
               layer_norm='input',
               name='transformer_tower'):
    """Initializes TransformerTower.

    Args:
      value_size: dimensionality of values per-head.
      num_heads: number of attention heads.
      num_layers: number of transformer blocks, where each block contains a
        multi-head attention layer and an MLP.
      causal: if True, applies a causal mask.
      key_size: optional dimensionality of key size. If unspecified then it is
        set to `value_size`.
      shared_attention: if True, attention params are shared across all layers.
      output_size: if set, the desired output dimensionality. By default the
        output size is `value_size` x `num_heads`.
      mlp_hidden_sizes: tuple containing dimensionality of mlp layer(s). If
        multiple values are specified, the mlp contains multiple layers for each
        transformer block.
      dropout_rate: dropout rate applied to hidden activations, attention, and
        positional encodings.
      use_relative_positions: if False, applies absolute positional encodings.
        If true, uses relative positional encodings from Dai et al. 2019.
      clamp_time_range: clamps max temporal positional encoding if specified.
      same_attention_length: if True, attention is masked to ensure each
        position in the sequence contains the same length of attention.
      layer_norm: Where to apply layer-norm in Transformer block. Can be one of
        'input' (Vaswani et al. 2017), 'output', or 'both'.
      name: name of variable scope.
    """
    super(TransformerTower, self).__init__(name=name)
    self._causal = causal
    self._mask = None

    if key_size is None:
      key_size = value_size
    self._key_size = key_size
    self._value_size = value_size
    self._shared_attention = shared_attention
    self._num_heads = num_heads
    self._num_layers = num_layers
    self._output_size = output_size
    self._embedding_size = self._value_size * self._num_heads
    self._mlp_hidden_sizes = list(mlp_hidden_sizes) + [self._embedding_size]
    self._multihead_attention = None
    self._object_embeddings = None
    self._dropout_rate = dropout_rate
    self._positional_encodings = None
    self._use_relative_positions = use_relative_positions
    self._clamp_time_range = clamp_time_range
    self._same_attention_length = same_attention_length
    self._layer_norm = layer_norm
    self._attention_modules = []
    self._object_mlps = []

  def get_sublayers(self, is_training):
    if self._multihead_attention is None or not self._shared_attention:
      attention_module = MultiheadAttention(
          value_size=self._value_size,
          key_size=self._key_size,
          num_heads=self._num_heads,
          mask=self._mask,
          positional_encodings=self._positional_encodings,
          use_relative_positions=self._use_relative_positions,
          init_std=2. / np.sqrt(self._num_layers),
      )
      self._multihead_attention = ResidualDropoutWrapper(
          attention_module, self._dropout_rate, layer_norm=self._layer_norm)
    mlp = default_mlp(
        self._mlp_hidden_sizes, init_std=2. / np.sqrt(self._num_layers))
    object_mlp = ResidualDropoutWrapper(
        mlp, self._dropout_rate, layer_norm=self._layer_norm)

    self._attention_modules.append(attention_module)
    self._object_mlps.append(mlp)
    return self._multihead_attention, object_mlp

  def _build(self, inputs, state=None, condition=None, is_training=True):
    """Calculates multi-layer self attention and mlp transformation.

    Args:
      inputs: Tensor of shape [batch_size, num_steps, dim_size].
      state: optional tensor of shape [batch_size, memory_size, dim_size].
      condition: optional tensor to condition on. The shape is shape
        [batch_size, dim_size].
      is_training: If true, dropout is applied.

    Returns:
      output: tensor of shape [batch_size, num_steps, output_dim_size].
      state: list of length `num_layers` containing AttentionState tuples.
    """

    # inputs: [B, N, F]
    if condition is not None:
      condition_tile = tf.tile(
          tf.expand_dims(condition, 1), [1, tf.shape(inputs)[1], 1])
      inputs = tf.concat([inputs, condition_tile], -1)

    if state is None:
      memory_sizes = [0]
    elif isinstance(state[0], CompressedMemoryState):
      cm_mem_size = max(_memory_size(s.compressed_memory) for s in state)
      em_mem_size = max(_memory_size(s.episodic_memory) for s in state)
      memory_sizes = [cm_mem_size, em_mem_size]
    else:
      memory_sizes = [max([_memory_size(s) for s in state])]
    chunk_size = inputs.get_shape().as_list()[1]
    self._positional_encodings = []
    # Creates positional encodings for different memory types.
    for i, memory_size in enumerate(memory_sizes):
      seq_len = chunk_size + memory_size
      key_positions = get_position_encodings(
          sequence_length=seq_len,
          hidden_size=inputs.get_shape().as_list()[2],
          clamp_value=self._clamp_time_range,
      )
      if is_training:
        key_positions = tf.nn.dropout(key_positions, rate=self._dropout_rate)
      key_positions = tf.cast(key_positions, dtype=inputs.dtype)
      query_positions = key_positions[:, -chunk_size:, :]
      self._positional_encodings.append((key_positions, query_positions))

    if self._causal:
      self._mask = create_mask(inputs, state, self._same_attention_length)

    layer_i_inputs = inputs
    attention_states = []
    for i in range(self._num_layers):
      with tf.variable_scope('layer_%d' % i, reuse=tf.AUTO_REUSE):
        multihead_attention, object_mlp = self.get_sublayers(is_training)
        # Multihead attention with residuals.
        state_i = None if state is None else state[i]
        attention_outputs, attention_state = multihead_attention(
            layer_i_inputs,
            state=state_i,
            is_training=is_training,
            dropout_keep_prob=1. - self._dropout_rate)
        attention_states.append(attention_state)
        # Feed-forward with residuals.
        output = object_mlp(
            attention_outputs,
            is_training=is_training,
            dropout_keep_prob=1 - self._dropout_rate)
        layer_i_inputs = output

    if self._output_size is not None:
      output = basic.BatchApply(
          basic.Linear(self._output_size, use_bias=False))(
              output)

    return output, attention_states

  def attention_module(self, i):
    """Returns the i-th layer attention module."""
    return self._attention_modules[i]


class TransformerXL(rnn_core.RNNCore):
  """Transformer with memory of past activations.

  From Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
  dai et al. 2019, https://arxiv.org/abs/1901.02860.

  The TransformerXL can be used in two modes:

  * batched, i.e. when chunk_size > 0. Here the model expects 3D input of size
    `[batch_size, chunk_size, input_dim]`. In practice, the input chunk size
    can be of varying (but statically defined) shape.
  * single-step, i.e. when chunk_size = 0. Here the model expects 2D input
    `[batch_size, input_dim]`.

  """

  def __init__(self,
               core_config,
               memory_size,
               chunk_size,
               name='transformer_xl'):
    """Constructs TransformerXL graph.

    Args:
      core_config: dictionary with TransformerTower config.
      memory_size: size of memory.
      chunk_size: expected chunk size of inputs, if greater than zero inputs are
        of size [batch_size, chunk_size, input_dim]. If equal to zero inputs are
        of size [batch_size, input_dim].
      name: name of variable scope.
    """

    super(TransformerXL, self).__init__(name=name)
    self._core_config = core_config
    self._memory_size = memory_size
    self._chunk_size = chunk_size

    # Extract some size information from the core config.
    self._num_layers = self._core_config['num_layers']
    self._value_size = self._core_config['value_size']
    self._key_size = self._core_config.get('key_size') or self._value_size
    self._num_heads = self._core_config['num_heads']
    self._dropout_rate = self._core_config.get('dropout_rate', 0.)
    self._embedding_size = self._num_heads * self._value_size

  def _build(self, inputs, prev_state, is_training=True):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) == 2:
      inputs = tf.expand_dims(inputs, 1)
      chunk_size = 1
    else:
      _, chunk_size, _ = input_shape
    inputs = default_mlp([self._embedding_size], activate_final=True)(
        inputs,
        is_training=is_training,
        dropout_keep_prob=1 - self._dropout_rate)

    transformer = TransformerTower(**self._core_config)
    state_for_transformer = None if self._memory_size == 0 else prev_state
    output, attention_state = transformer(
        inputs, state=state_for_transformer, is_training=is_training)

    next_state = []
    for i, state_i in enumerate(prev_state):
      # Append new elements to memory.
      attn_state_i = attention_state[i]
      memory = tf.concat([state_i, attn_state_i.embeddings], 1)[:, chunk_size:]
      next_state.append(memory)

    if self._chunk_size == 0:  # For the use-case as a single-step RNN.
      output = tf.squeeze(output, 1)

    return output, next_state

  @property
  def state_size(self):
    memory_shape = tf.TensorShape([self._memory_size, self._embedding_size])
    return [memory_shape] * self._num_layers

  @property
  def output_size(self):
    if self._chunk_size == 0:
      return tf.TensorShape([self._embedding_size])
    else:
      return tf.TensorShape([self._chunk_size, self._embedding_size])


class PoolCompressor(base.AbstractModule):
  """Compress sequence using simple pooling."""

  def __init__(self,
               compression_rate=2,
               kernel_size=0,
               pooling='AVG',
               compressed_memory_size=None,
               episodic_memory_size=None,
               name='pool_compressor'):
    """Instantiates compression module.

    Args:
      compression_rate: integer >= 1. The memory will be compressed from T
        time-steps to T / compression_rate. In implementation, this corresponds
        to the stride size of the module.
      kernel_size: The additional kernel size, if we wish for overlapping
        compressed vectors. The total conv1d kernel size is 'compression_rate +
        kernel_size', (as the stride is 'compression_rate').
      pooling: AVG or MAX pooling.
      compressed_memory_size: Size of compressed memory store.
      episodic_memory_size: Size of episodic memory store.
      name: module name, used for variable scoping.
    """
    super(PoolCompressor, self).__init__(name=name)
    self._compression_rate = compression_rate
    self._compressed_memory_size = compressed_memory_size
    self._episodic_memory_size = episodic_memory_size
    self._pooling = pooling
    self._kernel_size = kernel_size

  def _build(self, memory, **unused_kwargs):
    pooled_memories = tf.nn.pool(
        memory,
        window_shape=(self._compression_rate + self._kernel_size,),
        data_format='NWC',
        strides=(self._compression_rate,),
        padding='VALID',
        pooling_type=self._pooling)
    return pooled_memories, tf.zeros([], dtype=memory.dtype)


class ConvCompressor(base.AbstractModule):
  """Compress sequence using convolutions, with respect to a desired loss."""

  def __init__(self,
               compression_rate,
               compressed_memory_size,
               episodic_memory_size,
               kernel_size=0,
               dilation_rates=None,
               loss='mha',
               name='conv_compressor'):
    """Instantiates convolutional compression module.

    Args:
      compression_rate: integer >= 1. The memory will be compressed from T
        time-steps to T / compression_rate. In implementation, this corresponds
        to the stride size of the module.
      compressed_memory_size: Size of compressed memory store.
      episodic_memory_size: Size of regular memory equiv. to TXL's memory.
      kernel_size: The additional kernel size, if we wish for overlapping
        compressed vectors. The total conv1d kernel size is 'compression_rate +
        kernel_size', (as the stride is 'compression_rate').
      dilation_rates: optional iterable of dilation rates for deep dilated
        convnet, e.g. [1, 2, 4].
      loss: Either 'ae' for an auto-encoder compression loss, or 'mha' for a
        multi-head attention loss. The multi-head attention loss attempts to
        reconstruct the attention outputs between the (sequence, memory) and
        (sequence, compressed_memory).
      name: module name, used for variable scoping.
    """
    super(ConvCompressor, self).__init__(name=name)
    self._loss = loss
    self._stride = compression_rate
    self._kernel_size = kernel_size
    self._dilation_rates = dilation_rates
    self._compressed_memory_size = compressed_memory_size
    self._episodic_memory_size = episodic_memory_size

  def _build(self,
             memory,
             attention_state=None,
             attention_module=None,
             is_training=False,
             dropout_keep_prob=0.5):
    """Builds graph to compress memory and return auxiliary loss.

    Args:
      memory: [batch, chunk_size, hidden_size] tensor to be compressed.
      attention_state: AttentionState named tuple containing the queries, keys,
        and values that were computed at a given layer.
      attention_module: the attention module (sonnet class). Useful for
        accessing the multi-head attention sub-modules, used to transform hidden
        states into queries, keys, and values.
      is_training: if is training, useful for dropout gating.
      dropout_keep_prob: the probability of dropout. Currently unused!

    Returns:
      (compressed_memory, loss) tuple. Compressed_memory is of size
        [batch, time / compression_rate, hidden_size]. The loss is a scalar.
    """
    _, chunk_size, hidden_size = memory.get_shape().as_list()
    # Start of queryable sequence, from sequence of hiddens. If the memory is
    # larger than the chunk size, the whole sequence will be used for queries.
    seq_s = max(chunk_size - self._episodic_memory_size, 0)

    memory = tf.stop_gradient(memory)
    compressed_memory = memory
    if self._dilation_rates is not None:
      for rate in self._dilation_rates:
        conv = snt_conv.Conv1D(
            hidden_size,
            kernel_shape=2,
            rate=rate,
            use_bias=False,
            padding=snt_conv.VALID,
            name='conv_rate_%d' % rate,
        )
        compressed_memory = conv(compressed_memory)
        compressed_memory = tf.nn.relu(compressed_memory)

    conv = snt_conv.Conv1D(
        hidden_size,
        kernel_shape=self._stride + self._kernel_size,
        stride=self._stride,
        use_bias=False,
        padding=snt_conv.VALID,
    )
    # We stop gradients for the compression inputs. This is to avoid the network
    # shaping them to be compressible. We would like to compress them
    # *conditioned* on the task-specific representations that are learned.

    # Queries from current sequence.
    queries = tf.stop_gradient(attention_state.queries[:, :, seq_s:])
    # Memory of past hidden activations to be compressed.
    compressed_memory = conv(compressed_memory)
    if self._loss == 'ae':
      transpose_conv = conv.transpose()
      recovered_memory = transpose_conv(compressed_memory)
      loss = tf.reduce_mean(tf.square(recovered_memory - memory))
    elif self._loss == 'mha':
      # We share the attention module's parameters, but we stop gradients from
      # flowing to these parameters with respect to the auxiliary loss, as we
      # don't want the attention module to shape queries, keys, and values to
      # be compressible.
      stop_gradient_getter = custom_getters.Context(
          custom_getters.stop_gradient)
      with stop_gradient_getter:
        # Calculates attention from sequence over memory.
        memory_keys = attention_module.multihead_linear(memory, name='key')
        memory_values = attention_module.multihead_linear(memory, name='value')
        read_words_with_memory = simple_attention(queries, memory_keys,
                                                  memory_values)

        # Calculates attention from sequence over compressed memory.
        compressed_keys = attention_module.multihead_linear(
            compressed_memory, name='key')
        compressed_values = attention_module.multihead_linear(
            compressed_memory, name='value')
        read_words_with_compressed_memory = simple_attention(
            queries, compressed_keys, compressed_values)

      loss = tf.reduce_mean(
          tf.square(read_words_with_memory - read_words_with_compressed_memory))
    else:
      raise NotImplementedError(
          'Unrecognised loss: %r, expected `ae` or `mha`' % self._loss)
    return compressed_memory, loss


def _compute_avg_attention(attention_state,
                           compressed_memory_size,
                           episodic_memory_size,
                           chunk_size,
                           n_buckets=6):
  """Computes average attention for Compressive Transformer.

  Computes average attention for `n_buckets` over the sequence,
  episodic memory, and compressed memory. In total there are 3 x n_buckets.

  Args:
    attention_state: An AttentionState object.
    compressed_memory_size: scalar size of compressed memory.
    episodic_memory_size: scalar size of episodic memory.
    chunk_size: size of input sequence.
    n_buckets: number of buckets to average attention per memory,
      compressed memory, and sequence.

  Returns:
    Tuple of (names, avg_weights) where each is a list. The names are
      <segment_type>_<bucket_id>, i.e. cm_0, cm_1, em_0, em_1, seq_0, seq_1.
      The bucket index is ordered by time, higher values are for attention
      over more recent buckets of [seq/cm/em]. The avg_weights are the list
      of corresponding values.

  """
  cm_size = compressed_memory_size
  em_size = episodic_memory_size
  split_sizes = []
  split_names = []
  if cm_size > 0:
    split_sizes += [int(cm_size / n_buckets)] * (n_buckets - 1)
    split_sizes += [cm_size - int(cm_size / n_buckets) * (n_buckets - 1)]
    split_names += ['cm_p%d' % i for i in range(n_buckets)]
  if em_size > 0:
    split_sizes += [int(em_size / n_buckets)] * (n_buckets - 1)
    split_sizes += [em_size - int(em_size / n_buckets) * (n_buckets - 1)]
    split_names += ['em_p%d' % i for i in range(n_buckets)]

  split_sizes += [int(chunk_size / n_buckets)] * (n_buckets - 1)
  split_sizes += [chunk_size - int(chunk_size / n_buckets) * (n_buckets - 1)]

  split_names += ['seq_p%d' % i for i in range(n_buckets)]
  avg_weights = tf.reduce_mean(attention_state.weights, axis=[0, 1, 2])
  split_avg_weights = tf.split(avg_weights, split_sizes)
  split_avg_weights = [tf.reduce_sum(x) for x in split_avg_weights]
  return split_names, split_avg_weights


class CompressiveTransformer(rnn_core.RNNCore):
  """Transformer with compressive memory.

  From "Compressive Transformers for Long-Range Sequence Modelling"
  Rae et al. 2019, https://arxiv.org/abs/1911.05507

  """

  def __init__(self,
               core_config,
               chunk_size,
               episodic_memory_size,
               compressed_memory_size,
               compression_rate=2,
               compression_ctor=ConvCompressor,
               compression_config=None,
               export_stats=False,
               name='compressive_transformer'):
    """Constructs Compressive Transformer.

    Wraps a TransformerTower and includes a slot-based memory (like the
    TransformerXL) alongside a compressed memory which is populated from
    the oldest slot-based memories, passed through a compression network.
    To train the compression network, an auxiliary compression loss is
    added to the collection 'auxiliary_losses'.

    Args:
      core_config: dictionary with TransformerTower config.
      chunk_size: expected chunk size of inputs, if greater than zero inputs are
        of size [batch_size, chunk_size, input_dim]. If equal to zero inputs are
        of size [batch_size, input_dim].
      episodic_memory_size: size of slot-based memory (i.e. TransformerXL mem).
      compressed_memory_size: size of compressed memory. Total attention len is
        episodic_memory_size + compressed_memory_size + chunk_size.
      compression_rate: Factor of compression from episodic memories to
        compressed memories, i.e. `2` means M memories are mapped to M/2
        compressed memories.
      compression_ctor: Constructor of compression network, e.g. ConvCompressor,
        PoolCompressor, or any newly specified network.
      compression_config: optional dictionary with keyword arguments for
        compression network.
      export_stats: exports compression loss and attention weight per layer to a
        tf collection 'stats_export' if true. Can slow down training.
      name: name of variable scope.
    """

    super(CompressiveTransformer, self).__init__(name=name)
    self._core_config = core_config
    self._episodic_memory_size = episodic_memory_size
    self._compressed_memory_size = compressed_memory_size
    self._chunk_size = chunk_size
    self._compression_config = dict(compression_config or [])
    self._compression_rate = compression_rate
    self._compression_config.update({
        'compression_rate': compression_rate,
        'compressed_memory_size': self._compressed_memory_size,
        'episodic_memory_size': self._episodic_memory_size,
    })
    self._compression_ctor = compression_ctor
    self._export_stats = export_stats

    # Extract some size information from the core config.
    self._num_layers = self._core_config['num_layers']
    self._value_size = self._core_config['value_size']
    self._key_size = self._core_config.get('key_size') or self._value_size
    self._num_heads = self._core_config['num_heads']
    self._dropout_rate = self._core_config.get('dropout_rate', 0.)
    self._embedding_size = self._num_heads * self._value_size

  def _build(self, inputs, prev_state, is_training=True):
    """Builds graph.

    Args:
      inputs: 3D tensor of shape [batch_size, chunk_size, input_dim] or
        2D tensor of shape [batch_size, input_dim].
      prev_state: list of length `num_layers` containing `CompressedMemoryState`
        tuples.
      is_training: applies dropout if true.

    Returns:
      output: tensor equal in rank to `inputs` with final dimension equal to
        `embedding_size` = `key_size` * `num_heads`.
      next_state: list of length `num_layers` containing `CompressedMemoryState`
        tuples.
    """
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) == 2:
      inputs = tf.expand_dims(inputs, 1)

    _, chunk_size, _ = inputs.get_shape().as_list()
    num_layers_t = tf.constant(self._num_layers, dtype=inputs.dtype)

    inputs = default_mlp([self._embedding_size], activate_final=True)(
        inputs,
        is_training=is_training,
        dropout_keep_prob=1 - self._dropout_rate)
    transformer = TransformerTower(**self._core_config)
    state_for_transformer = (None
                             if self._episodic_memory_size == 0 else prev_state)
    output, attention_state = transformer(
        inputs, state=state_for_transformer, is_training=is_training)

    min_num_to_compress = (
        self._compression_rate + self._compression_config.get('kernel_size', 0))
    num_to_compress = min(max(min_num_to_compress, chunk_size),
                          chunk_size + self._episodic_memory_size - 1)

    def apply_compression_generic(attn_state, attn_module, mem_to_compress,
                                  prev_compressed_memory):
      """Instantiates compression module and returns fn to build graph."""
      compress_module = self._compression_ctor(**self._compression_config)

      def _inner_fn():
        """Returns (updated compressed memory, compression loss)."""
        next_compressed_memory, compression_loss = compress_module(
            mem_to_compress,
            attention_state=attn_state,
            attention_module=attn_module,
            is_training=is_training,
            dropout_keep_prob=1 - self._dropout_rate,
        )
        compressed_memory, _ = _concat_and_slice(prev_compressed_memory,
                                                 next_compressed_memory)
        return compressed_memory, compression_loss

      return _inner_fn

    def dont_apply_compression_generic(prev_compressed_memory):
      """Instantiates fn to build dummy graph that skips any compression."""

      def _inner_fn():
        return (prev_compressed_memory,
                tf.zeros([], dtype=prev_compressed_memory.dtype))

      return _inner_fn

    next_state = []
    compression_loss = tf.zeros([], dtype=inputs.dtype)
    global_attention_weights = []
    stats_export_dict = {}
    for i, state_i in enumerate(prev_state):
      # Append new elements to memory.
      attn_state_i = attention_state[i]
      memory, concat_memory = _concat_and_slice(state_i.episodic_memory,
                                                attn_state_i.embeddings)

      sequence_index = state_i.index[0]
      # We special-case chunk_size=1, which is useful for sampling. In the
      # single time-step setting we only compress the memory every
      # 'compression_rate' steps. Otherwise we assume chunk_size is a multiple
      # of `compression_rate`, and thus multiple compressions can be performed
      # in parallel.
      to_compress = tf.logical_or(
          chunk_size > 1,
          tf.equal(sequence_index % self._compression_rate,
                   self._compression_rate - 1))[0]

      apply_compression_fn = apply_compression_generic(
          attn_state=attn_state_i,
          attn_module=transformer.attention_module(i),
          mem_to_compress=concat_memory[:, :num_to_compress],
          prev_compressed_memory=state_i.compressed_memory,
      )
      dont_apply_compression_fn = dont_apply_compression_generic(
          prev_compressed_memory=state_i.compressed_memory)

      compression_output = tf.cond(to_compress, apply_compression_fn,
                                   dont_apply_compression_fn)
      compressed_memory, compression_loss_i = compression_output
      compression_loss += compression_loss_i

      # Log useful stats, compression loss per layer.
      stats_export_dict['compression_loss_l%02d' % i] = compression_loss_i
      # Attention weights per layer.
      attn_names, attn_weights = _compute_avg_attention(
          attn_state_i, self._compressed_memory_size,
          self._episodic_memory_size, chunk_size)
      attn_names_i = [name + '_l%02d' % i for name in attn_names]
      stats_export_dict.update(dict(zip(attn_names_i, attn_weights)))

      # Avg global attention weights.
      if i == 0:
        global_attention_weights = [y / num_layers_t for y in attn_weights]
      else:
        global_attention_weights = [
            (x + y / num_layers_t)
            for x, y in zip(global_attention_weights, attn_weights)
        ]

      next_state.append(
          CompressedMemoryState(
              index=state_i.index + 1,
              episodic_memory=memory,
              compressed_memory=compressed_memory))

    next_state = tuple(next_state)
    compression_loss /= num_layers_t
    stats_export_dict.update(dict(zip(attn_names, global_attention_weights)))
    if is_training:
      tf.add_to_collections('auxiliary_losses', compression_loss)
    if self._export_stats:
      tf.add_to_collections('stats_export', stats_export_dict)

    if self._chunk_size == 0:  # For the use-case as a single-step RNN.
      output = tf.squeeze(output, 1)

    return output, next_state

  @property
  def state_size(self):
    memory_shape = tf.TensorShape(
        [self._episodic_memory_size, self._embedding_size])
    cm_shape = tf.TensorShape(
        [self._compressed_memory_size, self._embedding_size])
    index_shape = tf.TensorShape([1])
    shape_per_layer = CompressedMemoryState(
        index=index_shape,
        episodic_memory=memory_shape,
        compressed_memory=cm_shape)
    return tuple([shape_per_layer] * self._num_layers)

  @property
  def output_size(self):
    if self._chunk_size == 0:
      return tf.TensorShape([self._embedding_size])
    else:
      return tf.TensorShape([self._chunk_size, self._embedding_size])
