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

"""Modules for embedding integer ids."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# Dependency imports
from sonnet.python.modules import base
from sonnet.python.modules import util
import tensorflow as tf


def _embedding_dim(vocab_size):
  """Calculate a reasonable embedding size for a vocabulary.

  Rule of thumb is 6 * 4th root of vocab_size.

  Args:
    vocab_size: Size of the input vocabulary.
  Returns:
    The embedding size to use.
  Raises:
    ValueError: if `vocab_size` is invalid.
  """
  if not vocab_size or (vocab_size <= 0):
    raise ValueError("Invalid vocab_size %g." % vocab_size)
  return int(round(6.0 * math.sqrt(math.sqrt(vocab_size))))


class Embed(base.AbstractModule):
  """Module for embedding tokens in a low-dimensional space."""

  EMBEDDINGS = "embeddings"
  POSSIBLE_INITIALIZER_KEYS = {EMBEDDINGS}

  def __init__(self,
               vocab_size=None,
               embed_dim=None,
               existing_vocab=None,
               densify_gradients=False,
               initializers=None,
               partitioners=None,
               regularizers=None,
               trainable=True,
               custom_getter=None,
               name="embed"):
    """Constructs an Embed module.

    Args:
      vocab_size: int. Number of unique tokens to embed. If not provided, an
        existing vocabulary matrix from which vocab_size can be inferred must
        be provided as existing_vocab.
      embed_dim: int or None. Number of dimensions to assign to each embedding.
        If not specified, a sensible default is chosen based on `vocab_size`. If
        an existing vocabulary matrix initializes the module, this should not be
        provided as it will be inferred.
      existing_vocab: a [vocab_size, embed_dim] vocabulary matrix. Will be
        converted to a tf.float32 tensor. If provided, neither or vocab_size or
        embed_dim should be provided as they are inferred.
      densify_gradients: if True, we convert the embedding gradient from an
        indexed-slices to a regular tensor before sending it back to the
        parameter server. This avoids excess computation on the parameter
        server. Use this option for moderately sized embeddings, e.g.,
        a vocabulary size on the order of up to thousands. For embeddings larger
        than these, e.g. a vocabulary size on the order of tens or hundreds of
        thousands, set this to False.
      initializers: Optional dict containing initializers for embeddings (with
        key 'embeddings'). As a default, embeddings are initialized via a
        truncated normal distribution.
      partitioners: Optional dict containing partitioners for embeddings (with
        key 'embeddings'). As a default, no partitioners are used.
      regularizers: Optional dict containing regularizers for embeddings (with
        key 'embeddings'). As a default, no regularizers are used. A regularizer
        should be a function that takes a single `Tensor` as an input and
        returns a scalar `Tensor` output, e.g. the L1 and L2 regularizers
        in `tf.contrib.layers`.
      trainable: if True, the embeddings will be updated during training. If
        False, they are fixed to their initial values. If `trainable=False` and
        a regularizer is given, the resulting loss stays constant.
      custom_getter: Callable or dictionary of callables to use as
        custom getters inside the module. If a dictionary, the keys
        correspond to regexes to match variable names. See the `tf.get_variable`
        documentation for information about the custom_getter API.
      name: string. Name for this module.

    Raises:
      ValueError: if neither one of vocab_size or existing_vocab is provided, or
        if existing_vocab is provided along with vocab_size, embedding_dim,
        initializers, partitioners or regularizers (as these should
        be inferred).
    """
    if vocab_size is None and existing_vocab is None:
      raise ValueError("Must provide on of vocab_size or existing_vocab.")

    if existing_vocab is not None and not all(
        x is None for x in [vocab_size, embed_dim, initializers, partitioners]):
      raise ValueError("If existing_vocab is provided, none of vocab_size, "
                       "embedding_dim, initializers, or partitioners is "
                       "needed.")

    super(Embed, self).__init__(custom_getter=custom_getter, name=name)
    self._existing_vocab = None
    if existing_vocab is None:
      self._vocab_size = vocab_size
      self._embed_dim = embed_dim or _embedding_dim(self._vocab_size)
    else:
      self._existing_vocab = tf.convert_to_tensor(
          existing_vocab, dtype=tf.float32)
      existing_vocab_shape = self._existing_vocab.get_shape().with_rank(2)
      existing_vocab_shape.assert_is_fully_defined()
      self._vocab_size, self._embed_dim = existing_vocab_shape.as_list()

    self._initializers = util.check_initializers(
        initializers, self.POSSIBLE_INITIALIZER_KEYS)
    self._partitioners = util.check_partitioners(
        partitioners, self.POSSIBLE_INITIALIZER_KEYS)
    self._regularizers = util.check_regularizers(
        regularizers, self.POSSIBLE_INITIALIZER_KEYS)
    self._trainable = trainable
    self._densify_gradients = densify_gradients

  def _build(self, ids):
    """Lookup embeddings.

    Looks up an embedding vector for each value in `ids`. All ids must be within
    [0, vocab_size), else an `InvalidArgumentError` is raised at runtime.

    Args:
      ids: Tensor of dtype int64.

    Returns:
      Tensor of tf.shape(ids) + [embedding_dim] and dtype float32.
    """
    # Construct embeddings.
    if self._existing_vocab is None:
      if self.EMBEDDINGS not in self._initializers:
        self._initializers[self.EMBEDDINGS] = tf.initializers.random_normal()
      self._embeddings = tf.get_variable(
          "embeddings",
          shape=[self._vocab_size, self._embed_dim],
          dtype=tf.float32,
          initializer=self._initializers[self.EMBEDDINGS],
          partitioner=self._partitioners.get(self.EMBEDDINGS, None),
          regularizer=self._regularizers.get(self.EMBEDDINGS, None),
          trainable=self._trainable)
    else:
      self._embeddings = tf.get_variable(
          "embeddings",
          dtype=tf.float32,
          initializer=self._existing_vocab,
          regularizer=self._regularizers.get(self.EMBEDDINGS, None),
          trainable=self._trainable)

    if self._densify_gradients:
      # On the backwards pass, we convert the gradient from indexed-slices to a
      # regular tensor before sending it back to the parameter server.
      # This avoids excess computation on the parameter server.
      # In eager mode we do not need the conversion.
      # Add a check whether we are in eager mode when it is supported.
      embeddings = util.convert_gradient_to_tensor(self._embeddings)
    else:
      embeddings = self._embeddings

    # Lookup embeddings
    return tf.nn.embedding_lookup(embeddings, ids, name="embedding_lookup")

  @property
  def vocab_size(self):
    """Size of input vocabulary."""
    return self._vocab_size

  @property
  def embed_dim(self):
    """Size of embedding vectors."""
    return self._embed_dim

  @property
  def embeddings(self):
    """Returns the Variable containing embeddings.

    Returns:
      A 2D Variable containing one embedding vector per row, constructed in the
        most recent __call__.

    Raises:
      base.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._embeddings
