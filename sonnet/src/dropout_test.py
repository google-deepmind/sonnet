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
"""Tests for sonnet.v2.src.dropout."""

from absl.testing import parameterized
import numpy as np
from sonnet.src import dropout
from sonnet.src import test_utils
import tensorflow as tf


class DropoutTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(np.arange(.0, .9, .1))
  def test_sum_close(self, rate):
    mod = dropout.Dropout(rate=rate)
    x = tf.ones([1000])
    rtol = 0.3 if "TPU" in self.device_types else 0.1
    self.assertAllClose(
        tf.reduce_sum(mod(x, is_training=True)),
        tf.reduce_sum(mod(x, is_training=False)),
        rtol=rtol)

  @parameterized.parameters(np.arange(0, .9, .1))
  def test_dropout_rate(self, rate):
    mod = dropout.Dropout(rate=rate)
    x = tf.ones([1000])
    x = mod(x, is_training=True)

    # We should have dropped something, test we're within 10% of rate.
    # (or 30% on a TPU)
    rtol = 0.3 if "TPU" in self.device_types else 0.1
    kept = tf.math.count_nonzero(x).numpy()
    keep_prob = 1 - rate
    self.assertAllClose(kept, 1000 * keep_prob, rtol=rtol)

  def test_dropout_is_actually_random(self):
    mod = dropout.Dropout(rate=0.5)
    x = tf.ones([1000])
    tf.random.set_seed(1)
    y1 = mod(x, is_training=True)
    y2 = mod(x, is_training=True)
    self.assertNotAllClose(y1, y2)

  @parameterized.parameters(True, False)
  def test_with_tf_function_with_booleans(self, autograph):
    """tf.function compilation correctly handles if statement."""

    layer = dropout.Dropout(rate=0.5)
    layer = tf.function(layer, autograph=autograph)

    inputs = tf.ones([2, 5, 3, 3, 3])
    expected = tf.zeros_like(inputs)

    for is_training in (True, False):
      outputs = layer(inputs, is_training)
      self.assertEqual(outputs.shape, expected.shape)

  @parameterized.parameters(True, False)
  def test_with_tf_function_with_variables(self, autograph):
    """tf.function correctly handles if statement when argument is Variable."""

    layer = dropout.Dropout(rate=0.5)
    layer = tf.function(layer, autograph=autograph)

    inputs = tf.ones([2, 5, 3, 3, 3])
    expected = tf.zeros_like(inputs)
    is_training_variable = tf.Variable(False, trainable=False)

    for is_training in (True, False):
      is_training_variable.assign(is_training)
      outputs = layer(inputs, is_training_variable)
      self.assertEqual(outputs.shape, expected.shape)


if __name__ == "__main__":
  tf.test.main()
