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

"""Tests for sonnet.v2.src.sgd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from sonnet.src import sgd
from sonnet.src import test_utils
import tensorflow as tf


class SGDTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(sgd.SGD, sgd.FastSGD)
  def testDense(self, sgd_class):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    optimizer = sgd_class(learning_rate=3.)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-14., -13.], [-6., -5.]],
                        [x.numpy() for x in parameters])

  @parameterized.parameters(sgd.SGD, sgd.FastSGD)
  def testSparse(self, sgd_class):
    if self.primary_device == "TPU":
      self.skipTest("IndexedSlices not supported on TPU.")

    parameters = [tf.Variable([[1.], [2.]]), tf.Variable([[3.], [4.]])]
    updates = [tf.IndexedSlices(tf.constant([0.1], shape=[1, 1]),
                                tf.constant([0]), tf.constant([2, 1])),
               tf.IndexedSlices(tf.constant([0.01], shape=[1, 1]),
                                tf.constant([1]), tf.constant([2, 1]))]
    optimizer = sgd_class(learning_rate=3.)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[1.0 - 3.0 * 0.1], [2.0]], parameters[0].numpy())
    self.assertAllClose([[3.0], [4.0 - 3.0 * 0.01]], parameters[1].numpy())

  @parameterized.parameters(sgd.SGD, sgd.FastSGD)
  def testNoneUpdate(self, sgd_class):
    parameters = [tf.Variable([1., 2.])]
    updates = [None]
    optimizer = sgd_class(learning_rate=3.)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[1., 2.]], [x.numpy() for x in parameters])

  @parameterized.parameters(sgd.SGD, sgd.FastSGD)
  def testVariableLearningRate(self, sgd_class):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    learning_rate = tf.Variable(3.)
    optimizer = sgd_class(learning_rate=learning_rate)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-14., -13.], [-6., -5.]],
                        [x.numpy() for x in parameters])
    learning_rate.assign_sub(1.)
    self.assertEqual(2., optimizer.learning_rate.numpy())
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-24., -23.], [-12., -11.]],
                        [x.numpy() for x in parameters])

  @parameterized.parameters(sgd.SGD, sgd.FastSGD)
  def testLearningRateDTypeConversion(self, sgd_class):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    dtype = tf.int32 if self.primary_device == "TPU" else tf.int64
    learning_rate = tf.Variable(3, dtype=dtype)
    optimizer = sgd_class(learning_rate=learning_rate)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-14., -13.], [-6., -5.]],
                        [x.numpy() for x in parameters])

  @parameterized.parameters(sgd.SGD, sgd.FastSGD)
  def testDifferentLengthUpdatesParams(self, sgd_class):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.])]
    optimizer = sgd_class(learning_rate=3.)
    with self.assertRaisesRegexp(
        ValueError, "`updates` and `parameters` must be the same length."):
      optimizer.apply(updates, parameters)

  @parameterized.parameters(sgd.SGD, sgd.FastSGD)
  def testEmptyParams(self, sgd_class):
    optimizer = sgd_class(learning_rate=3.)
    with self.assertRaisesRegexp(ValueError, "`parameters` cannot be empty."):
      optimizer.apply([], [])

  @parameterized.parameters(sgd.SGD, sgd.FastSGD)
  def testInconsistentDTypes(self, sgd_class):
    parameters = [tf.Variable([1., 2.], name="param0")]
    updates = [tf.constant([5, 5])]
    optimizer = sgd_class(learning_rate=3.)
    with self.assertRaisesRegexp(
        ValueError, "DType of .* is not equal to that of parameter .*param0.*"):
      optimizer.apply(updates, parameters)

if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
