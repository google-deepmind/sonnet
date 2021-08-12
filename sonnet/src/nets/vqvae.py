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

from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import moving_averages
from sonnet.src import types

import tensorflow as tf


class VectorQuantizer(base.Module):
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

  Attributes:
    embedding_dim: integer representing the dimensionality of the tensors in the
      quantized space. Inputs to the modules must be in this format as well.
    num_embeddings: integer, the number of vectors in the quantized space.
    commitment_cost: scalar which controls the weighting of the loss terms (see
      equation 4 in the paper - this variable is Beta).
  """

  def __init__(self,
               embedding_dim: int,
               num_embeddings: int,
               commitment_cost: types.FloatLike,
               dtype: tf.DType = tf.float32,
               name: str = 'vector_quantizer'):
    """Initializes a VQ-VAE module.

    Args:
      embedding_dim: dimensionality of the tensors in the quantized space.
        Inputs to the modules must be in this format as well.
      num_embeddings: number of vectors in the quantized space.
      commitment_cost: scalar which controls the weighting of the loss terms
        (see equation 4 in the paper - this variable is Beta).
      dtype: dtype for the embeddings variable, defaults to tf.float32.
      name: name of the module.
    """
    super().__init__(name=name)
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost

    embedding_shape = [embedding_dim, num_embeddings]
    initializer = initializers.VarianceScaling(distribution='uniform')
    self.embeddings = tf.Variable(
        initializer(embedding_shape, dtype), name='embeddings')

  def __call__(self, inputs, is_training):
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
    flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

    distances = (
        tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
        2 * tf.matmul(flat_inputs, self.embeddings) +
        tf.reduce_sum(self.embeddings**2, 0, keepdims=True))

    encoding_indices = tf.argmax(-distances, 1)
    encodings = tf.one_hot(encoding_indices,
                           self.num_embeddings,
                           dtype=distances.dtype)

    # NB: if your code crashes with a reshape error on the line below about a
    # Tensor containing the wrong number of values, then the most likely cause
    # is that the input passed in does not have a final dimension equal to
    # self.embedding_dim. Ideally we would catch this with an Assert but that
    # creates various other problems related to device placement / TPUs.
    encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
    quantized = self.quantize(encoding_indices)

    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs)**2)
    q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs))**2)
    loss = q_latent_loss + self.commitment_cost * e_latent_loss

    # Straight Through Estimator
    quantized = inputs + tf.stop_gradient(quantized - inputs)
    avg_probs = tf.reduce_mean(encodings, 0)
    perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                       tf.math.log(avg_probs + 1e-10)))

    return {
        'quantize': quantized,
        'loss': loss,
        'perplexity': perplexity,
        'encodings': encodings,
        'encoding_indices': encoding_indices,
        'distances': distances,
    }

  def quantize(self, encoding_indices):
    """Returns embedding tensor for a batch of indices."""
    w = tf.transpose(self.embeddings, [1, 0])
    # TODO(mareynolds) in V1 we had a validate_indices kwarg, this is no longer
    # supported in V2. Are we missing anything here?
    return tf.nn.embedding_lookup(w, encoding_indices)


class VectorQuantizerEMA(base.Module):
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

  Attributes:
    embedding_dim: integer representing the dimensionality of the tensors in the
      quantized space. Inputs to the modules must be in this format as well.
    num_embeddings: integer, the number of vectors in the quantized space.
    commitment_cost: scalar which controls the weighting of the loss terms (see
      equation 4 in the paper).
    decay: float, decay for the moving averages.
    epsilon: small float constant to avoid numerical instability.
  """

  def __init__(self,
               embedding_dim,
               num_embeddings,
               commitment_cost,
               decay,
               epsilon=1e-5,
               dtype=tf.float32,
               name='vector_quantizer_ema'):
    """Initializes a VQ-VAE EMA module.

    Args:
      embedding_dim: integer representing the dimensionality of the tensors in
        the quantized space. Inputs to the modules must be in this format as
        well.
      num_embeddings: integer, the number of vectors in the quantized space.
      commitment_cost: scalar which controls the weighting of the loss terms
        (see equation 4 in the paper - this variable is Beta).
      decay: float between 0 and 1, controls the speed of the Exponential Moving
        Averages.
      epsilon: small constant to aid numerical stability, default 1e-5.
      dtype: dtype for the embeddings variable, defaults to tf.float32.
      name: name of the module.
    """
    super().__init__(name=name)
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    if not 0 <= decay <= 1:
      raise ValueError('decay must be in range [0, 1]')
    self.decay = decay
    self.commitment_cost = commitment_cost
    self.epsilon = epsilon

    embedding_shape = [embedding_dim, num_embeddings]
    initializer = initializers.VarianceScaling(distribution='uniform')
    self.embeddings = tf.Variable(
        initializer(embedding_shape, dtype), trainable=False, name='embeddings')

    self.ema_cluster_size = moving_averages.ExponentialMovingAverage(
        decay=self.decay, name='ema_cluster_size')
    self.ema_cluster_size.initialize(tf.zeros([num_embeddings], dtype=dtype))

    self.ema_dw = moving_averages.ExponentialMovingAverage(
        decay=self.decay, name='ema_dw')
    self.ema_dw.initialize(self.embeddings)

  def __call__(self, inputs, is_training):
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
    flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

    distances = (
        tf.reduce_sum(flat_inputs**2, 1, keepdims=True) -
        2 * tf.matmul(flat_inputs, self.embeddings) +
        tf.reduce_sum(self.embeddings**2, 0, keepdims=True))

    encoding_indices = tf.argmax(-distances, 1)
    encodings = tf.one_hot(encoding_indices,
                           self.num_embeddings,
                           dtype=distances.dtype)

    # NB: if your code crashes with a reshape error on the line below about a
    # Tensor containing the wrong number of values, then the most likely cause
    # is that the input passed in does not have a final dimension equal to
    # self.embedding_dim. Ideally we would catch this with an Assert but that
    # creates various other problems related to device placement / TPUs.
    encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
    quantized = self.quantize(encoding_indices)
    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs)**2)

    if is_training:
      updated_ema_cluster_size = self.ema_cluster_size(
          tf.reduce_sum(encodings, axis=0))

      dw = tf.matmul(flat_inputs, encodings, transpose_a=True)
      updated_ema_dw = self.ema_dw(dw)

      n = tf.reduce_sum(updated_ema_cluster_size)
      updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                  (n + self.num_embeddings * self.epsilon) * n)

      normalised_updated_ema_w = (
          updated_ema_dw / tf.reshape(updated_ema_cluster_size, [1, -1]))

      self.embeddings.assign(normalised_updated_ema_w)
      loss = self.commitment_cost * e_latent_loss

    else:
      loss = self.commitment_cost * e_latent_loss

    # Straight Through Estimator
    quantized = inputs + tf.stop_gradient(quantized - inputs)
    avg_probs = tf.reduce_mean(encodings, 0)
    perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                       tf.math.log(avg_probs + 1e-10)))

    return {
        'quantize': quantized,
        'loss': loss,
        'perplexity': perplexity,
        'encodings': encodings,
        'encoding_indices': encoding_indices,
        'distances': distances,
    }

  def quantize(self, encoding_indices):
    """Returns embedding tensor for a batch of indices."""
    w = tf.transpose(self.embeddings, [1, 0])
    return tf.nn.embedding_lookup(w, encoding_indices)
