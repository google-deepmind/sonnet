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

from sonnet.src.optimizers import optimizer_tests
from sonnet.src.optimizers import sgd
import tensorflow as tf


class SGDTest(optimizer_tests.OptimizerTestBase):

  def make_optimizer(self, *args, **kwargs):
    if "learning_rate" not in kwargs:
      kwargs["learning_rate"] = 3.
    return sgd.SGD(*args, **kwargs)

  def testDense(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    optimizer = self.make_optimizer(learning_rate=3.)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-14., -13.], [-6., -5.]],
                        [x.numpy() for x in parameters])

  def testSparse(self):
    if self.primary_device == "TPU":
      self.skipTest("IndexedSlices not supported on TPU.")

    parameters = [tf.Variable([[1.], [2.]]), tf.Variable([[3.], [4.]])]
    updates = [
        tf.IndexedSlices(
            tf.constant([0.1], shape=[1, 1]), tf.constant([0]),
            tf.constant([2, 1])),
        tf.IndexedSlices(
            tf.constant([0.01], shape=[1, 1]), tf.constant([1]),
            tf.constant([2, 1]))
    ]
    optimizer = self.make_optimizer(learning_rate=3.)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[1.0 - 3.0 * 0.1], [2.0]], parameters[0].numpy())
    self.assertAllClose([[3.0], [4.0 - 3.0 * 0.01]], parameters[1].numpy())

  def testVariableLearningRate(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    learning_rate = tf.Variable(3.)
    optimizer = self.make_optimizer(learning_rate=learning_rate)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-14., -13.], [-6., -5.]],
                        [x.numpy() for x in parameters])
    learning_rate.assign_sub(1.)
    self.assertEqual(2., optimizer.learning_rate.numpy())
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-24., -23.], [-12., -11.]],
                        [x.numpy() for x in parameters])

  def testLearningRateDTypeConversion(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    dtype = tf.int32 if self.primary_device == "TPU" else tf.int64
    learning_rate = tf.Variable(3, dtype=dtype)
    optimizer = self.make_optimizer(learning_rate=learning_rate)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-14., -13.], [-6., -5.]],
                        [x.numpy() for x in parameters])


class ReferenceSGDTest(SGDTest):

  def make_optimizer(self, *args, **kwargs):
    if "learning_rate" not in kwargs:
      kwargs["learning_rate"] = 3.
    if hasattr(tf.keras.optimizers, "legacy"):
      return optimizer_tests.WrappedTFOptimizer(
          tf.keras.optimizers.legacy.SGD(**kwargs))
    return optimizer_tests.WrappedTFOptimizer(tf.keras.optimizers.SGD(**kwargs))


if __name__ == "__main__":
  tf.test.main()
