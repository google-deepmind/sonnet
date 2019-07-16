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

"""Common tests for Sonnet optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet.src import test_utils
import tensorflow as tf


class OptimizerTestBase(test_utils.TestCase):
  """Common tests for Sonnet optimizers."""

  def make_optimizer(self, *args, **kwargs):
    raise NotImplementedError()

  def testNoneUpdate(self):
    parameters = [tf.Variable(1.), tf.Variable(2.)]
    updates = [None, tf.constant(3.)]
    optimizer = self.make_optimizer()
    optimizer.apply(updates, parameters)
    self.assertAllClose(1., parameters[0].numpy())

  def testDifferentLengthUpdatesParams(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.])]
    optimizer = self.make_optimizer()
    with self.assertRaisesRegexp(
        ValueError, "`updates` and `parameters` must be the same length."):
      optimizer.apply(updates, parameters)

  def testEmptyParams(self):
    optimizer = self.make_optimizer()
    with self.assertRaisesRegexp(ValueError, "`parameters` cannot be empty."):
      optimizer.apply([], [])

  def testAllUpdatesNone(self):
    parameters = [tf.Variable(1.), tf.Variable(2.)]
    updates = [None, None]
    optimizer = self.make_optimizer()
    with self.assertRaisesRegexp(
        ValueError, "No updates provided for any parameter"):
      optimizer.apply(updates, parameters)

  def testInconsistentDTypes(self):
    parameters = [tf.Variable([1., 2.], name="param0")]
    updates = [tf.constant([5, 5])]
    optimizer = self.make_optimizer()
    with self.assertRaisesRegexp(
        ValueError, "DType of .* is not equal to that of parameter .*param0.*"):
      optimizer.apply(updates, parameters)

  def testUnsuppportedStrategyError(self):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      parameters = [tf.Variable(1.0)]
      updates = [tf.constant(0.1)]
      optimizer = self.make_optimizer()
    with self.assertRaisesRegexp(
        ValueError,
        "Sonnet optimizers are not compatible with `MirroredStrategy`"):
      strategy.experimental_run_v2(lambda: optimizer.apply(updates, parameters))

if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
