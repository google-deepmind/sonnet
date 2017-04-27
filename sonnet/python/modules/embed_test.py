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

"""Tests for sonnet.python.modules.embed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import sonnet as snt
import tensorflow as tf

from tensorflow.python.ops import variables


class EmbedTest(tf.test.TestCase):

  def setUp(self):
    super(EmbedTest, self).setUp()
    self._batch_size = 3
    self._vocab_size = 7
    self._embed_dim = 1
    self._embed_mod = snt.Embed(
        vocab_size=self._vocab_size, embed_dim=self._embed_dim)
    self._ids = np.asarray([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])

  def testOutputType(self):
    # Output shape should be same as ids, except with a full embedding for each
    # value.
    embeddings = self._embed_mod(tf.convert_to_tensor(self._ids))
    expected_shape = list(self._ids.shape) + [self._embed_dim]
    self.assertTrue(embeddings.get_shape().is_compatible_with(expected_shape))
    self.assertEqual(embeddings.dtype, tf.float32)

  def testComputation(self):
    # Initialize each embedding to its index. Thus, the lookup ids are the same
    # as the embeddings themselves.
    initializers = {"embeddings": tf.constant_initializer(
        [[0], [1], [2], [3], [4], [5], [6]], dtype=tf.float32)}
    embed_mod = snt.Embed(
        vocab_size=self._vocab_size,
        embed_dim=self._embed_dim,
        initializers=initializers)
    embeddings = embed_mod(tf.convert_to_tensor(self._ids))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      embeddings_ = sess.run(embeddings)
      expected_embeddings = np.reshape(
          self._ids, newshape=list(self._ids.shape) + [self._embed_dim])
      self.assertAllClose(embeddings_, expected_embeddings)

  def testVocabTooSmall(self):
    # If an index doesn't fit in the vocab, there will be no embedding for it
    # and an exception should be raised.
    ids = self._ids.copy()
    ids[0, 0] = self._vocab_size
    ids = tf.convert_to_tensor(ids)
    embeddings = self._embed_mod(ids)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(embeddings)

  def testNegativeIds(self):
    # Negative ids are not allowed.
    ids = self._ids.copy()
    ids[0, 0] = -1
    ids = tf.convert_to_tensor(ids)
    embeddings = self._embed_mod(ids)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(embeddings)

  def testDefaultVocabSize(self):
    embed_mod = snt.Embed(vocab_size=100, embed_dim=None, name="embed_small")
    self.assertEqual(embed_mod.embed_dim, 19)

    embed_mod = snt.Embed(vocab_size=1000000, embed_dim=None, name="embed_big")
    self.assertEqual(embed_mod.embed_dim, 190)

  def testInitializers(self):
    # Since all embeddings are initialized to zero, the extracted embeddings
    # should be as well.


    initializers = {"embeddings": tf.zeros_initializer()}
    embed_mod = snt.Embed(
        vocab_size=self._vocab_size,
        embed_dim=self._embed_dim,
        initializers=initializers)
    embeddings = embed_mod(tf.convert_to_tensor(self._ids))
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      embeddings_ = sess.run(embeddings)
      self.assertAllEqual(embeddings_, np.zeros_like(embeddings_))

  def testPartitioners(self):
    # Partition embeddings such that there's one variable per vocabulary entry.
    partitioners = {"embeddings": tf.variable_axis_size_partitioner(
        4 * self._embed_dim)}
    embed_mod = snt.Embed(
        vocab_size=self._vocab_size,
        embed_dim=self._embed_dim,
        partitioners=partitioners)
    embeddings = embed_mod(tf.convert_to_tensor(self._ids))
    self.assertEqual(type(embed_mod.embeddings), variables.PartitionedVariable)
    self.assertEqual(len(embed_mod.embeddings), self._vocab_size)

    # Ensure that tf.nn.embedding_lookup() plays nicely with embedding
    # variables.
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(embeddings)

  def testInvalidRegularizationParameters(self):
    regularizer = tf.contrib.layers.l1_regularizer(scale=0.5)
    with self.assertRaisesRegexp(KeyError, "Invalid regularizer keys.*"):
      snt.Embed(
          vocab_size=self._vocab_size,
          embed_dim=self._embed_dim,
          regularizers={"not_embeddings": regularizer})

    err = "Regularizer for 'embeddings' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.Embed(vocab_size=self._vocab_size,
                embed_dim=self._embed_dim,
                regularizers={"embeddings": tf.zeros([1, 2, 3])})

  def testRegularizersInRegularizationLosses(self):
    regularizer = tf.contrib.layers.l1_regularizer(scale=0.5)
    embed = snt.Embed(
        vocab_size=self._vocab_size,
        embed_dim=self._embed_dim,
        regularizers={"embeddings": regularizer})
    embed(tf.convert_to_tensor(self._ids))

    regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertRegexpMatches(regularizers[0].name, ".*l1_regularizer.*")

  def testProperties(self):
    self.assertEqual(self._embed_mod.vocab_size, self._vocab_size)
    self.assertEqual(self._embed_mod.embed_dim, self._embed_dim)

    # Embeddings aren't accessible until module is connected to a graph.
    with self.assertRaises(snt.NotConnectedError):
      _ = self._embed_mod.embeddings
    self._embed_mod(tf.convert_to_tensor(self._ids))
    self.assertEqual(type(self._embed_mod.embeddings), tf.Variable)

  def testExistingVocab(self):
    # Check that the module can be initialised with an existing vocabulary.
    existing = np.array(
        [[1, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int32)
    expected = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int32)
    true_vocab_size, true_embed_dim = existing.shape

    inputs = tf.constant(np.array([0, 2, 1]), dtype=tf.int32)
    embed_mod = snt.Embed(existing_vocab=existing)
    embeddings = embed_mod(inputs)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      embeddings_ = sess.run(embeddings)
      self.assertAllClose(embeddings_, expected)
      self.assertEqual(embed_mod.vocab_size, true_vocab_size)
      self.assertEqual(embed_mod.embed_dim, true_embed_dim)

if __name__ == "__main__":
  tf.test.main()
