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

from absl import logging
from sonnet.src import replicator
from sonnet.src import test_utils
import tensorflow as tf


def replicator_all_devices(device_type):
  # NOTE: The explicit device list is required since currently Replicator
  # only considers CPU and GPU devices. This means on TPU by default we only
  # mirror on the local CPU.
  devices = tf.config.experimental.list_logical_devices(device_type=device_type)
  devices = [d.name for d in devices]
  logging.info("Replicating over %s", devices)
  return replicator.Replicator(devices=devices)


class ReplicatorTest(test_utils.TestCase):

  def test_scope(self):
    strategy = replicator_all_devices(self.primary_device)
    with strategy.scope():
      v = tf.Variable(1.0)
    self.assertEqual(
        tf.VariableSynchronization.ON_READ, v.primary.synchronization)
    self.assertEqual(tf.VariableAggregation.ONLY_FIRST_REPLICA, v.aggregation)
    self.assertTrue(v.trainable)

  def test_experimental_run_v2(self):
    strategy = replicator_all_devices(self.primary_device)
    def step():
      v = tf.Variable(1.0)
      self.assertEqual(
          tf.VariableSynchronization.ON_READ, v.primary.synchronization)
      self.assertEqual(tf.VariableAggregation.ONLY_FIRST_REPLICA, v.aggregation)
      self.assertTrue(v.trainable)
    strategy.experimental_run_v2(step)

if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
