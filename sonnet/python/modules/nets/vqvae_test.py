# pylint: disable=g-bad-file-header
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
"""Tests for sonnet.python.modules.nets.vqvae."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import sonnet as snt
import tensorflow as tf


class VqvaeTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (snt.nets.VectorQuantizer,
       {'embedding_dim': 4, 'num_embeddings': 8,
        'commitment_cost': 0.25}),
      (snt.nets.VectorQuantizerEMA,
       {'embedding_dim': 6, 'num_embeddings': 13,
        'commitment_cost': 0.5, 'decay': 0.1})
  )
  def testConstruct(self, constructor, kwargs):
    vqvae = constructor(**kwargs)
    # Batch of input vectors to quantize
    inputs_np = np.random.randn(16, kwargs['embedding_dim']).astype(np.float32)
    inputs = tf.constant(inputs_np)

    # Set is_training to False, otherwise for the EMA case just evaluating the
    # forward pass will change the embeddings, meaning that some of our computed
    # closest embeddings will be incorrect.
    vq_output = vqvae(inputs, is_training=False)

    # Output shape is correct
    self.assertEqual(vq_output['quantize'].shape, inputs.shape)

    init_op = tf.global_variables_initializer()
    with self.test_session() as session:
      session.run(init_op)
      vq_output_np, embeddings_np = session.run([vq_output, vqvae.embeddings])

    self.assertEqual(embeddings_np.shape, (kwargs['embedding_dim'],
                                           kwargs['num_embeddings']))

    # Check that each input was assigned to the embedding it is closest to.
    distances = ((inputs_np ** 2).sum(axis=1, keepdims=True)
                 - 2 * np.dot(inputs_np, embeddings_np)
                 + (embeddings_np**2).sum(axis=0, keepdims=True))
    closest_index = np.argmax(-distances, axis=1)
    self.assertAllEqual(closest_index,
                        np.argmax(vq_output_np['encodings'], axis=1))

  @parameterized.parameters(
      (snt.nets.VectorQuantizer,
       {'embedding_dim': 4, 'num_embeddings': 8,
        'commitment_cost': 0.25}),
      (snt.nets.VectorQuantizerEMA,
       {'embedding_dim': 6, 'num_embeddings': 13,
        'commitment_cost': 0.5, 'decay': 0.1})
  )
  def testShapeChecking(self, constructor, kwargs):
    vqvae = constructor(**kwargs)
    wrong_shape_input = np.random.randn(100, kwargs['embedding_dim'] * 2)
    with self.assertRaisesRegexp(ValueError, 'Cannot reshape a tensor'):
      vqvae(tf.constant(wrong_shape_input.astype(np.float32)),
            is_training=False)

  def testEmaUpdating(self):
    embedding_dim = 6
    vqvae = snt.nets.VectorQuantizerEMA(
        embedding_dim=embedding_dim, num_embeddings=7,
        commitment_cost=0.5, decay=0.1)

    batch_size = 16
    input_ph = tf.placeholder(shape=[batch_size, embedding_dim],
                              dtype=tf.float32)
    output = vqvae(input_ph, is_training=True)
    embeddings = vqvae.embeddings

    init_op = tf.global_variables_initializer()
    with self.test_session() as session:
      session.run(init_op)
      # embedding should change every time we put some data through, even though
      # we are not passing any gradients through.
      prev_w = session.run(embeddings)
      for _ in range(10):
        session.run(output, {input_ph: np.random.randn(batch_size,
                                                       embedding_dim)})
        current_w = session.run(embeddings)
        self.assertFalse((prev_w == current_w).all())
        prev_w = current_w


if __name__ == '__main__':
  tf.test.main()
