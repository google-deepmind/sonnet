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

"""Tests for sonnet.v2.src.adam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from sonnet.src import adam
from sonnet.src import test_utils
import tensorflow as tf


class AdamTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(adam.Adam, adam.ReferenceAdam)
  def testDense(self, opt_class):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    optimizer = opt_class(learning_rate=0.001)
    # Step 1 of Adam
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.999, 1.999], [2.999, 3.999]],
                        [x.numpy() for x in parameters])
    # Step 2 of Adam
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.998, 1.998], [2.998, 3.998]],
                        [x.numpy() for x in parameters])
    # Step 3 of Adam
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.997, 1.997], [2.997, 3.997]],
                        [x.numpy() for x in parameters])

  @parameterized.parameters(adam.Adam, adam.ReferenceAdam)
  def testNoneUpdate(self, opt_class):
    parameters = [tf.Variable([1., 2.])]
    updates = [None]
    optimizer = opt_class(learning_rate=0.001)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[1., 2.]], [x.numpy() for x in parameters])

  @parameterized.parameters(adam.Adam, adam.ReferenceAdam)
  def testVariableHyperParams(self, opt_class):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    learning_rate = tf.Variable(0.001)
    optimizer = opt_class(learning_rate=learning_rate)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.999, 1.999], [2.999, 3.999]],
                        [x.numpy() for x in parameters])
    learning_rate.assign(0.1)
    self.assertAlmostEqual(0.1, optimizer.learning_rate.numpy())
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.899, 1.899], [2.899, 3.899]],
                        [x.numpy() for x in parameters], rtol=1e-4)

  @parameterized.parameters(adam.Adam, adam.ReferenceAdam)
  def testHyperParamDTypeConversion(self, opt_class):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    dtype = tf.float32 if self.primary_device == "TPU" else tf.float64
    learning_rate = tf.Variable(0.001, dtype=dtype)
    beta1 = tf.Variable(0.9, dtype=dtype)
    beta2 = tf.Variable(0.999, dtype=dtype)
    epsilon = tf.Variable(1e-8, dtype=dtype)
    optimizer = opt_class(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.999, 1.999], [2.999, 3.999]],
                        [x.numpy() for x in parameters], rtol=1e-4)

  @parameterized.parameters(adam.Adam, adam.ReferenceAdam)
  def testDifferentLengthUpdatesParams(self, opt_class):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.])]
    optimizer = opt_class(learning_rate=0.001)
    with self.assertRaisesRegexp(
        ValueError, "`updates` and `parameters` must be the same length."):
      optimizer.apply(updates, parameters)

  @parameterized.parameters(adam.Adam, adam.ReferenceAdam)
  def testEmptyParams(self, opt_class):
    optimizer = opt_class(learning_rate=0.001)
    with self.assertRaisesRegexp(ValueError, "`parameters` cannot be empty."):
      optimizer.apply([], [])

  @parameterized.parameters(adam.Adam, adam.ReferenceAdam)
  def testInconsistentDTypes(self, opt_class):
    parameters = [tf.Variable([1., 2.], name="param0")]
    updates = [tf.constant([5, 5])]
    optimizer = opt_class(learning_rate=0.001)
    with self.assertRaisesRegexp(
        ValueError, "DType of .* is not equal to that of parameter .*param0.*"):
      optimizer.apply(updates, parameters)

  @parameterized.parameters(adam.Adam, adam.ReferenceAdam)
  def testMomentVariablesColocatedWithOriginal(self, opt_class):
    optimizer = opt_class(learning_rate=0.001)
    with tf.device("CPU:0"):
      var = tf.Variable(1.0)
    optimizer.apply([tf.constant(0.1)], [var])
    self.assertEqual(optimizer.m[0].device, var.device)
    self.assertEqual(optimizer.v[0].device, var.device)

if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
