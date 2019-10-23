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
# from __future__ import google_type_annotations
from __future__ import print_function

from sonnet.src import base
from sonnet.src import test_utils
import tensorflow as tf


class WrappedTFOptimizer(base.Optimizer):
  """Wraps a TF optimizer in the Sonnet API."""

  wrapped = None

  def __init__(self, optimizer: tf.optimizers.Optimizer):
    super(WrappedTFOptimizer, self).__init__()
    self.wrapped = optimizer

  def __getattr__(self, name):
    return getattr(self.wrapped, name)

  def apply(self, updates, params):
    self.wrapped.apply_gradients(zip(updates, params))


def is_tf_optimizer(optimizer):
  return isinstance(optimizer, WrappedTFOptimizer)


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
    optimizer = self.make_optimizer()
    if is_tf_optimizer(optimizer):
      self.skipTest("TF optimizers don't check the lenghs of params/updates.")

    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.])]
    with self.assertRaisesRegexp(
        ValueError, "`updates` and `parameters` must be the same length."):
      optimizer.apply(updates, parameters)

  def testEmptyParams(self):
    optimizer = self.make_optimizer()
    if is_tf_optimizer(optimizer):
      self.skipTest("TF optimizers don't error on empty params.")

    with self.assertRaisesRegexp(ValueError, "`parameters` cannot be empty."):
      optimizer.apply([], [])

  def testAllUpdatesNone(self):
    parameters = [tf.Variable(1.), tf.Variable(2.)]
    updates = [None, None]
    optimizer = self.make_optimizer()
    if is_tf_optimizer(optimizer):
      msg = "No gradients provided for any variable"
    else:
      msg = "No updates provided for any parameter"
    with self.assertRaisesRegexp(ValueError, msg):
      optimizer.apply(updates, parameters)

  def testInconsistentDTypes(self):
    optimizer = self.make_optimizer()
    if is_tf_optimizer(optimizer):
      self.skipTest("TF optimizers raise a cryptic error message here.")

    parameters = [tf.Variable([1., 2.], name="param0")]
    updates = [tf.constant([5, 5])]
    with self.assertRaisesRegexp(
        ValueError, "DType of .* is not equal to that of parameter .*param0.*"):
      optimizer.apply(updates, parameters)

  def testUnsuppportedStrategyError(self):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      parameters = [tf.Variable(1.0)]
      updates = [tf.constant(0.1)]
      optimizer = self.make_optimizer()
      if is_tf_optimizer(optimizer):
        self.skipTest("TF optimizers aren't restricted to Sonnet strategies.")
    with self.assertRaisesRegexp(
        ValueError,
        "Sonnet optimizers are not compatible with `MirroredStrategy`"):
      strategy.experimental_run_v2(lambda: optimizer.apply(updates, parameters))


if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
