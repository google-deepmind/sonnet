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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl import logging
from absl.testing import parameterized
from sonnet.src import replicator as snt_replicator
from sonnet.src import test_utils
import tensorflow as tf


def replicator_all_devices(device_type):
  # TODO(petebu) Enable `Replicator` tests on TPU.
  if device_type == "TPU":
    logging.info("Using TpuReplicator")
    return snt_replicator.TpuReplicator()

  # NOTE: The explicit device list is required since currently Replicator
  # only considers CPU and GPU devices. This means on TPU by default we only
  # mirror on the local CPU.
  devices = tf.config.experimental.list_logical_devices(device_type=device_type)
  devices = [d.name for d in devices]
  logging.info("Replicating over %s", devices)
  return snt_replicator.Replicator(devices=devices)


def _create_variable_in_replicator_scope(replicator):
  with replicator.scope():
    v = tf.Variable(1.)
  return v


class TrainableVariable(object):

  def __call__(self):
    if not hasattr(self, "v"):
      self.v = tf.Variable(1.)
    return self.v


def _create_variable_in_step_fn(replicator):
  o = TrainableVariable()
  replicator.experimental_run_v2(o)
  return o.v


class ReplicatorTest(test_utils.TestCase, parameterized.TestCase):

  # Avoid running tests inside a `with tf.device("TPU:0"):` block.
  ENTER_PRIMARY_DEVICE = False

  @parameterized.parameters(
      [_create_variable_in_replicator_scope, _create_variable_in_step_fn])
  def test_variable_synchronization_default(self, create_var):
    replicator = replicator_all_devices(self.primary_device)
    v = create_var(replicator)
    self.assertEqual(
        tf.VariableSynchronization.ON_READ, v.primary.synchronization)

  @parameterized.parameters(
      [_create_variable_in_replicator_scope, _create_variable_in_step_fn])
  def test_variable_aggregation_default(self, create_var):
    replicator = replicator_all_devices(self.primary_device)
    v = create_var(replicator)
    self.assertEqual(tf.VariableAggregation.ONLY_FIRST_REPLICA, v.aggregation)

  @parameterized.parameters(
      [_create_variable_in_replicator_scope, _create_variable_in_step_fn])
  def test_variable_trainable_default(self, create_var):
    replicator = replicator_all_devices(self.primary_device)
    v = create_var(replicator)
    self.assertTrue(v.trainable)

  @parameterized.parameters([True, False])
  def test_variable_trainable(self, trainable):
    replicator = replicator_all_devices(self.primary_device)
    with replicator.scope():
      v = tf.Variable(1., trainable=trainable)
    self.assertEqual(trainable, v.trainable)

  @parameterized.parameters(
      *itertools.product(
          [("assign", 1.), ("assign_add", 1.), ("assign_sub", -1.)],
          [True, False]))
  def test_assign(self, updates, cross_replica):
    replicator = replicator_all_devices(self.primary_device)
    with replicator.scope():
      v = tf.Variable(0.)
    method_name, update_value = updates
    update_fn = lambda: getattr(v, method_name)(update_value)
    if cross_replica:
      # NOTE: Explicitly not running inside replicator.scope (fn should handle).
      update_fn()
    else:
      replicator.experimental_run_v2(update_fn)
    for component in v._values:
      self.assertAllEqual(component.read_value(), tf.ones_like(component))

  @parameterized.parameters(True, False)
  def test_read_value(self, cross_replica):
    replicator = replicator_all_devices(self.primary_device)
    with replicator.scope():
      v = tf.Variable(0.)
    if cross_replica:
      values = [v.read_value()]
    else:
      values = replicator.experimental_run_v2(v.read_value)
      values = replicator.experimental_local_results(values)
    for component in v._values:
      for value in values:
        self.assertAllEqual(component.read_value(), value)


def setUpModule():
  # If a physical GPU is available make sure TF sees at least two.
  gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
  if len(gpus) == 1:
    logging.info("Splitting one physical GPU into two logical GPUs.")
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
