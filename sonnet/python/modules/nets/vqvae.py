# Copyright 2018 The Sonnet Authors. All Rights Reserved.
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
"""Sonnet implementation of VQ-VAE https://arxiv.org/abs/1711.00937."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet.python.modules import base


import tensorflow as tf

from tensorflow.python.training import moving_averages


class VectorQuantizer(base.AbstractModule):
  """Sonnet module representing the VQ-VAE layer.

  Implements the algorithm presented in
  'Neural Discrete Representation Learning' by van den Oord et al.
  https://arxiv.org/abs/1711.00937

  Input any tensor to be quantized. Last dimension will be used as space in
  which to quantize. All other dimensions will be flattened and will be seen
  as different examples to quantize.

  The output tensor will have the same shape as the input.

  For example a tensor with shape [16, 32, 32, 64] will be reshaped into
  [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
  independently.

  Args:
    embedding_dim: integer representing the dimensionality of the tensors in the
      quantized space. Inputs to the modules must be in this format as well.
    num_embeddings: integer, the number of vectors in the quantized space.
    commitment_cost: scalar which controls the weighting of the loss terms
      (see equation 4 in the paper - this variable is Beta).
  """

  def __init__(self, embedding_dim, num_embeddings, commitment_cost,
               name='vq_layer'):
    super(VectorQuantizer, self).__init__(name=name)
    self._embedding_dim = embedding_dim
    self._num_embeddings = num_embeddings
    self._commitment_cost = commitment_cost

    with self._enter_variable_scope():
      initializer = tf.uniform_unit_scaling_initializer()
      self._w = tf.get_variable('embedding', [embedding_dim, num_embeddings],
                                initializer=initializer, trainable=True)

  def _build(self, inputs, is_training):
    """Connects the module to some inputs.

    Args:
      inputs: Tensor, final dimension must be equal to embedding_dim. All other
        leading dimensions will be flattened and treated as a large batch.
      is_training: boolean, whether this connection is to training data.

    Returns:
      dict containing the following keys and values:
        quantize: Tensor containing the quantized version of the input.
        loss: Tensor containing the loss to optimize.
        perplexity: Tensor containing the perplexity of the encodings.
        encodings: Tensor containing the discrete encodings, ie which element
          of the quantized space each input element was mapped to.
        encoding_indices: Tensor containing the discrete encoding indices, ie
          which element of the quantized space each input element was mapped to.
    """
    # Assert last dimension is same as self._embedding_dim
    input_shape = tf.shape(inputs)
    with tf.control_dependencies([
        tf.Assert(tf.equal(input_shape[-1], self._embedding_dim),
                  [input_shape])]):
      flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])

    distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                 - 2 * tf.matmul(flat_inputs, self._w)
                 + tf.reduce_sum(self._w ** 2, 0, keepdims=True))

    encoding_indices = tf.argmax(- distances, 1)
    encodings = tf.one_hot(encoding_indices, self._num_embeddings)
    encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
    quantized = self.quantize(encoding_indices)

    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
    q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
    loss = q_latent_loss + self._commitment_cost * e_latent_loss

    quantized = inputs + tf.stop_gradient(quantized - inputs)
    avg_probs = tf.reduce_mean(encodings, 0)
    perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))

    return {'quantize': quantized,
            'loss': loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices,}

  @property
  def embeddings(self):
    return self._w

  def quantize(self, encoding_indices):
    with tf.control_dependencies([encoding_indices]):
      w = tf.transpose(self.embeddings.read_value(), [1, 0])
    return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)


class VectorQuantizerEMA(base.AbstractModule):
  """Sonnet module representing the VQ-VAE layer.

  Implements a slightly modified version of the algorithm presented in
  'Neural Discrete Representation Learning' by van den Oord et al.
  https://arxiv.org/abs/1711.00937

  The difference between VectorQuantizerEMA and VectorQuantizer is that
  this module uses exponential moving averages to update the embedding vectors
  instead of an auxiliary loss. This has the advantage that the embedding
  updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
  ...) used for the encoder, decoder and other parts of the architecture. For
  most experiments the EMA version trains faster than the non-EMA version.

  Input any tensor to be quantized. Last dimension will be used as space in
  which to quantize. All other dimensions will be flattened and will be seen
  as different examples to quantize.

  The output tensor will have the same shape as the input.

  For example a tensor with shape [16, 32, 32, 64] will be reshaped into
  [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
  independently.

  Args:
    embedding_dim: integer representing the dimensionality of the tensors in the
      quantized space. Inputs to the modules must be in this format as well.
    num_embeddings: integer, the number of vectors in the quantized space.
    commitment_cost: scalar which controls the weighting of the loss terms (see
      equation 4 in the paper).
    decay: float, decay for the moving averages.
    epsilon: small float constant to avoid numerical instability.
  """

  def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay,
               epsilon=1e-5, name='VectorQuantizerEMA'):
    super(VectorQuantizerEMA, self).__init__(name=name)
    self._embedding_dim = embedding_dim
    self._num_embeddings = num_embeddings
    self._decay = decay
    self._commitment_cost = commitment_cost
    self._epsilon = epsilon

    with self._enter_variable_scope():
      initializer = tf.random_normal_initializer()
      # w is a matrix with an embedding in each column. When training, the
      # embedding is assigned to be the average of all inputs assigned to that
      # embedding.
      self._w = tf.get_variable(
          'embedding', [embedding_dim, num_embeddings],
          initializer=initializer, use_resource=True)
      self._ema_cluster_size = tf.get_variable(
          'ema_cluster_size', [num_embeddings],
          initializer=tf.constant_initializer(0), use_resource=True)
      self._ema_w = tf.get_variable(
          'ema_dw', initializer=self._w.initialized_value(), use_resource=True)

  def _build(self, inputs, is_training):
    """Connects the module to some inputs.

    Args:
      inputs: Tensor, final dimension must be equal to embedding_dim. All other
        leading dimensions will be flattened and treated as a large batch.
      is_training: boolean, whether this connection is to training data. When
        this is set to False, the internal moving average statistics will not be
        updated.

    Returns:
      dict containing the following keys and values:
        quantize: Tensor containing the quantized version of the input.
        loss: Tensor containing the loss to optimize.
        perplexity: Tensor containing the perplexity of the encodings.
        encodings: Tensor containing the discrete encodings, ie which element
          of the quantized space each input element was mapped to.
        encoding_indices: Tensor containing the discrete encoding indices, ie
          which element of the quantized space each input element was mapped to.
    """
    # Ensure that the weights are read fresh for each timestep, which otherwise
    # would not be guaranteed in an RNN setup. Note that this relies on inputs
    # having a data dependency with the output of the previous timestep - if
    # this is not the case, there is no way to serialize the order of weight
    # updates within the module, so explicit external dependencies must be used.
    with tf.control_dependencies([inputs]):
      w = self._w.read_value()
    input_shape = tf.shape(inputs)
    with tf.control_dependencies([
        tf.Assert(tf.equal(input_shape[-1], self._embedding_dim),
                  [input_shape])]):
      flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])

    distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                 - 2 * tf.matmul(flat_inputs, w)
                 + tf.reduce_sum(w ** 2, 0, keepdims=True))

    encoding_indices = tf.argmax(- distances, 1)
    encodings = tf.one_hot(encoding_indices, self._num_embeddings)
    encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
    quantized = self.quantize(encoding_indices)
    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)

    if is_training:
      updated_ema_cluster_size = moving_averages.assign_moving_average(
          self._ema_cluster_size, tf.reduce_sum(encodings, 0), self._decay)
      dw = tf.matmul(flat_inputs, encodings, transpose_a=True)
      updated_ema_w = moving_averages.assign_moving_average(self._ema_w, dw,
                                                            self._decay)
      n = tf.reduce_sum(updated_ema_cluster_size)
      updated_ema_cluster_size = (
          (updated_ema_cluster_size + self._epsilon)
          / (n + self._num_embeddings * self._epsilon) * n)

      normalised_updated_ema_w = (
          updated_ema_w / tf.reshape(updated_ema_cluster_size, [1, -1]))
      with tf.control_dependencies([e_latent_loss]):
        update_w = tf.assign(self._w, normalised_updated_ema_w)
        with tf.control_dependencies([update_w]):
          loss = self._commitment_cost * e_latent_loss

    else:
      loss = self._commitment_cost * e_latent_loss
    quantized = inputs + tf.stop_gradient(quantized - inputs)
    avg_probs = tf.reduce_mean(encodings, 0)
    perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))

    return {'quantize': quantized,
            'loss': loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices,}

  @property
  def embeddings(self):
    return self._w

  def quantize(self, encoding_indices):
    with tf.control_dependencies([encoding_indices]):
      w = tf.transpose(self.embeddings.read_value(), [1, 0])
    return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)
