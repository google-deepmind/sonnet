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
"""Tests for sonnet.v2.src.replicator."""

import os

from absl import logging
from absl.testing import parameterized
from sonnet.src import initializers
from sonnet.src import test_utils
from sonnet.src.distribute import replicator as snt_replicator
from sonnet.src.distribute import replicator_test_utils as replicator_utils
import tensorflow as tf


def _create_variable_in_cross_replica_context(replicator):
  with replicator.scope():
    v = tf.Variable(1.)
  return v


class TrainableVariable:

  def __call__(self):
    if not hasattr(self, "v"):
      self.v = tf.Variable(1.)
    return self.v


def _create_variable_in_replica_context(replicator):
  o = TrainableVariable()

  def create_var():
    replicator.run(o)

  # TpuReplicator doesn't support pure eager mode.
  if isinstance(replicator, snt_replicator.TpuReplicator):
    create_var = tf.function(create_var)

  create_var()
  return o.v


def all_variable_creators():
  return (("cross_replica_context", _create_variable_in_cross_replica_context),
          ("replica_context", _create_variable_in_replica_context))


class ReplicatorTest(test_utils.TestCase, parameterized.TestCase):

  # Avoid running tests inside a `with tf.device("TPU:0"):` block.
  ENTER_PRIMARY_DEVICE = False

  @test_utils.combined_named_parameters(replicator_utils.named_replicators(),
                                        all_variable_creators())
  def test_variable_synchronization_default(self, replicator_fn, create_var):
    replicator = replicator_fn()
    if replicator is None:
      self.skipTest("No replicator supplied.")
    v = create_var(replicator)
    self.assertEqual(tf.VariableSynchronization.ON_READ,
                     v.values[0].synchronization)

  @test_utils.combined_named_parameters(replicator_utils.named_replicators(),
                                        all_variable_creators())
  def test_variable_aggregation_default(self, replicator_fn, create_var):
    replicator = replicator_fn()
    if replicator is None:
      self.skipTest("No replicator supplied.")
    v = create_var(replicator)
    self.assertEqual(tf.VariableAggregation.ONLY_FIRST_REPLICA, v.aggregation)

  @test_utils.combined_named_parameters(replicator_utils.named_replicators(),
                                        all_variable_creators())
  def test_variable_trainable_default(self, replicator_fn, create_var):
    replicator = replicator_fn()
    if replicator is None:
      self.skipTest("No replicator supplied.")
    v = create_var(replicator)
    self.assertTrue(v.trainable)

  @test_utils.combined_named_parameters(replicator_utils.named_replicators(),
                                        test_utils.named_bools("trainable"))
  def test_variable_trainable(self, replicator_fn, trainable):
    replicator = replicator_fn()
    if replicator is None:
      self.skipTest("No replicator supplied.")
    with replicator.scope():
      v = tf.Variable(1., trainable=trainable)
    self.assertEqual(trainable, v.trainable)

  @test_utils.combined_named_parameters(replicator_utils.named_replicators(),
                                        (("assign", "assign", 1.),
                                         ("assign_add", "assign_add", 1.),
                                         ("assign_sub", "assign_sub", -1.)),
                                        test_utils.named_bools("cross_replica"))
  def test_assign(self, replicator_fn, method_name, value, cross_replica):
    replicator = replicator_fn()
    if replicator is None:
      self.skipTest("No replicator supplied.")

    with replicator.scope():
      v = tf.Variable(0.)
    update_fn = lambda: getattr(v, method_name)(value)
    if cross_replica:
      # NOTE: Explicitly not running inside replicator.scope (fn should handle).
      update_fn()
    else:
      # TpuReplicator doesn't support pure eager mode.
      if isinstance(replicator, snt_replicator.TpuReplicator):
        update_fn = tf.function(update_fn)
      replicator.run(update_fn)
    for component in v._values:
      self.assertAllEqual(component.read_value(), tf.ones_like(component))

  @test_utils.combined_named_parameters(replicator_utils.named_replicators(),
                                        test_utils.named_bools("cross_replica"))
  def test_read_value(self, replicator_fn, cross_replica):
    replicator = replicator_fn()
    if replicator is None:
      self.skipTest("No replicator supplied.")

    with replicator.scope():
      v = tf.Variable(0.)
    if cross_replica:
      values = [v.read_value()]
    else:
      # TpuReplicator doesn't support pure eager mode.
      if isinstance(replicator, snt_replicator.TpuReplicator):
        read_value_fn = tf.function(v.read_value)
      else:
        read_value_fn = v.read_value
      values = replicator.run(read_value_fn)
      values = replicator.experimental_local_results(values)
    for component in v._values:
      for value in values:
        self.assertAllEqual(component.read_value(), value)

  @parameterized.parameters(True, False)
  def test_falls_back_to_graph(self, autograph):
    if os.environ.get("GITHUB_ACTIONS", "") == "true" and autograph:
      self.skipTest("Autograph generated code has syntax error.")

    init = FailsInEagerMode()
    value = tf.function(
        snt_replicator.create_variables_eagerly(
            lambda: init([], tf.float32)), autograph=autograph)()
    self.assertEqual(value.numpy(), 1.)

  @parameterized.parameters(True, False)
  def test_requires_eager(self, autograph):
    init = MyOnesInitializer()
    value = tf.function(
        snt_replicator.create_variables_eagerly(
            lambda: init([], tf.float32)), autograph=autograph)()
    self.assertEqual(value.numpy(), 1.)

  @parameterized.parameters(True, False)
  def test_eager_variable_creator(self, autograph):
    variables = [None, None]
    eager_ones = tf.ones([])

    @snt_replicator.create_variables_eagerly
    def f():
      if variables[0] is None:
        graph_ones = tf.ones([])
        # NOTE: `graph_ones` will be resolved by `tf.get_static_value`.
        v1 = tf.Variable(graph_ones)
        v2 = tf.Variable(eager_ones)
        # Even though we're in a tf.function here, eager_variable_creator
        # should have popped us into an init_scope so we have eager variables.
        with tf.init_scope():
          self.assertEqual(v1.numpy(), 1.)
          self.assertEqual(v2.numpy(), 1.)
        variables[0] = v1
        variables[1] = v2

    tf.function(f, autograph=autograph)()


class MyOnesInitializer(initializers.Initializer):

  def __call__(self, shape, dtype):
    assert tf.executing_eagerly()
    return tf.ones(shape, dtype)


class FailsInEagerMode(initializers.Initializer):

  def __call__(self, shape, dtype):
    if tf.executing_eagerly():
      raise ValueError("Eager mode")
    return tf.ones(shape, dtype)


def setUpModule():
  # If a physical GPU is available make sure TF sees at least two.
  gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
  if len(gpus) == 1:
    logging.info("Splitting one physical GPU into two logical GPUs.")
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=1024),
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])


if __name__ == "__main__":
  tf.test.main()
