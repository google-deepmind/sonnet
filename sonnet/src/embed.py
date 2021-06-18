# Copyright 2019 The Sonnet Authors. All Rights Reserved.
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

"""Embedding module."""

import math
from typing import Optional

from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import types
import tensorflow as tf


class Embed(base.Module):
  """Module for embedding tokens in a low-dimensional space."""

  def __init__(self,
               vocab_size: Optional[int] = None,
               embed_dim: Optional[int] = None,
               existing_vocab: Optional[types.TensorLike] = None,
               densify_gradients: bool = False,
               initializer: Optional[initializers.Initializer] = None,
               trainable: bool = True,
               dtype: tf.DType = tf.float32,
               name: Optional[str] = None):
    """Constructs an Embed module.

    Args:
      vocab_size: Number of unique tokens to embed. If not provided, an
        existing vocabulary matrix from which vocab_size can be inferred must
        be provided as existing_vocab.
      embed_dim: Number of dimensions to assign to each embedding.
        If not specified, we use ``6 * sqrt(sqrt(vocab_size))``. If an existing
        vocabulary matrix initializes the module, this should not be provided as
        it will be inferred.
      existing_vocab: A ``[vocab_size, embed_dim]`` vocabulary matrix. Will be
        converted to a tf.float32 tensor. If provided, neither or vocab_size or
        embed_dim should be provided as they are inferred.
      densify_gradients: If True, we convert the embedding gradient from an
        ``tf.IndexedSlices`` to a regular tensor before sending it back to the
        parameter server. This avoids excess computation on the parameter
        server. Use this option for moderately sized embeddings, e.g.,
        a vocabulary size on the order of up to thousands. For embeddings larger
        than these, e.g. a vocabulary size on the order of tens or hundreds of
        thousands, set this to False.
      initializer: Initializer for the embeddings. By default,
        embeddings are initialized via a truncated normal distribution.
      trainable: if True, the embeddings will be updated during training. If
        False, they are fixed to their initial values.
      dtype: The dtype to use for the embedding. Defaults to float32.
      name: Name for this module.

    Raises:
      ValueError: if neither one of ``vocab_size`` or ``existing_vocab`` is
        provided, or if ``existing_vocab`` is provided along with
        ``vocab_size``, ``embedding_dim``, ``initializer`` (as these should be
        inferred).
    """
    super().__init__(name=name)

    if vocab_size is None and existing_vocab is None:
      raise ValueError("Must provide one of vocab_size or existing_vocab.")

    if existing_vocab is not None and (vocab_size or embed_dim or initializer):
      raise ValueError("If `existing_vocab` is provided, none of `vocab_size`, "
                       "`embedding_dim`, `initializer` are needed.")

    if existing_vocab is None:
      if embed_dim is None:
        embed_dim = embedding_dim(vocab_size)
      if initializer is None:
        initializer = initializers.TruncatedNormal()
      vocab = initializer([vocab_size, embed_dim], dtype)
    else:
      existing_vocab = tf.convert_to_tensor(existing_vocab, dtype=dtype)
      vocab_size, embed_dim = existing_vocab.shape
      vocab = existing_vocab

    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.densify_gradients = densify_gradients
    self.embeddings = tf.Variable(vocab, trainable=trainable, name="embeddings")

  def __call__(self, inputs):
    if self.densify_gradients:
      embeddings = dense_gradient(self.embeddings)
    else:
      embeddings = self.embeddings
    return tf.nn.embedding_lookup(embeddings, inputs)


def embedding_dim(vocab_size: int):
  """Calculate a reasonable embedding size for a vocabulary.

  Rule of thumb is ``6 * sqrt(sqrt(vocab_size))``.

  Args:
    vocab_size: Size of the input vocabulary.

  Returns:
    The embedding size to use.

  Raises:
    ValueError: if ``vocab_size`` is invalid.
  """
  if not vocab_size or (vocab_size <= 0):
    raise ValueError("Invalid vocab_size %g." % vocab_size)
  return int(round(6.0 * math.sqrt(math.sqrt(vocab_size))))


@tf.custom_gradient
def dense_gradient(x: tf.Tensor):
  """Identity operation whose gradient is converted to a ``tf.Tensor``.

  >>> embedding = tf.Variable(tf.random.normal([3, 3]))
  >>> with tf.GradientTape() as tape:
  ...   y = tf.nn.embedding_lookup(dense_gradient(embedding), [1])
  >>> tape.gradient(y, embedding).numpy()
  array([[ 0.,  0.,  0.],
         [ 1.,  1.,  1.],
         [ 0.,  0.,  0.]], dtype=float32)

  Args:
    x: A ``tf.Tensor``.

  Returns:
    The input ``tf.Tensor`` and a dense identity gradient function.
  """
  def grad(dy):
    if isinstance(dy, tf.IndexedSlices):
      return tf.convert_to_tensor(dy)
    else:
      return dy

  return x, grad
