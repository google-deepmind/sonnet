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
"""Tests for sonnet.v2.src.embed."""

from absl.testing import parameterized
from sonnet.src import embed
from sonnet.src import initializers
from sonnet.src import test_utils
import tensorflow as tf


class EmbedTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters([1, 10, 100])
  def test_vocab_size(self, vocab_size):
    e = embed.Embed(vocab_size=vocab_size)
    self.assertEqual(e.vocab_size, vocab_size)
    self.assertEqual(e.embeddings.shape[0], vocab_size)

  @parameterized.parameters([1, 10, 100])
  def test_embed_dim(self, embed_dim):
    e = embed.Embed(vocab_size=100, embed_dim=embed_dim)
    self.assertEqual(e.embed_dim, embed_dim)
    self.assertEqual(e.embeddings.shape[1], embed_dim)

  @parameterized.parameters([(1, 1), (10, 10), (100, 100)])
  def test_existing_vocab(self, vocab_size, embed_dim):
    existing_vocab = tf.ones([vocab_size, embed_dim])
    e = embed.Embed(existing_vocab=existing_vocab)
    self.assertEqual(e.vocab_size, vocab_size)
    self.assertEqual(e.embed_dim, embed_dim)
    self.assertAllEqual(e.embeddings.read_value(), existing_vocab)

  @parameterized.parameters([True, False])
  def test_densify_gradients(self, densify_gradients):
    e = embed.Embed(1, densify_gradients=densify_gradients)
    with tf.GradientTape() as tape:
      y = e([0])
      dy = tape.gradient(y, e.embeddings)
    if densify_gradients:
      self.assertIsInstance(dy, tf.Tensor)
    else:
      self.assertIsInstance(dy, tf.IndexedSlices)

  def test_initializer(self):
    e = embed.Embed(1, 1, initializer=initializers.Constant(28.))
    self.assertAllEqual(e.embeddings.read_value(), [[28.]])

  def test_pinned_to_cpu(self):
    with tf.device("CPU"):
      e = embed.Embed(1)
    spec = tf.DeviceSpec.from_string(e.embeddings.device)
    self.assertEqual(spec.device_type, "CPU")

  @parameterized.parameters([True, False])
  def test_trainable(self, trainable):
    e = embed.Embed(1, trainable=trainable)
    self.assertEqual(e.embeddings.trainable, trainable)

  @parameterized.parameters([tf.float32, tf.float16])
  def test_dtype(self, dtype):
    if dtype == tf.float16 and self.primary_device == "TPU":
      self.skipTest("float16 embeddings not supported on TPU.")
    e = embed.Embed(1, dtype=dtype)
    self.assertEqual(e.embeddings.dtype, dtype)

  def test_name(self):
    e = embed.Embed(1, name="my_embedding")
    self.assertEqual(e.name, "my_embedding")
    self.assertEqual(e.embeddings.name, "my_embedding/embeddings:0")


if __name__ == "__main__":
  tf.test.main()
