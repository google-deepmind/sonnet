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

"""Tests for sonnet.v2.src.optimizer_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
from sonnet.src import optimizer_utils
from sonnet.src import replicator as snt_replicator
from sonnet.src import test_utils
import tensorflow as tf


def maybe(cond, body):
  return body() if cond else None


def all_devices(device_type):
  devices = tf.config.experimental.list_logical_devices(device_type=device_type)
  return [d.name for d in devices]


MirroredStrategy = tf.distribute.MirroredStrategy
OneDeviceStrategy = tf.distribute.OneDeviceStrategy
TPUStrategy = tf.distribute.experimental.TPUStrategy
Replicator = snt_replicator.Replicator
TpuReplicator = snt_replicator.TpuReplicator
DTYPES = [tf.float32, tf.float16, tf.int32]


class OptimizerUtilsTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("one_device", lambda d: OneDeviceStrategy("/device:{}:0".format(d))),
      ("replicator", lambda d: Replicator(all_devices(d))),
      ("tpu_replicator", lambda d: maybe((d == "TPU"), TpuReplicator)))
  def test_check_strategy(self, replicator_fn):
    replicator = replicator_fn(self.primary_device)
    if replicator is None:
      self.skipTest("Skipping test.")

    with replicator.scope():
      optimizer_utils.check_strategy()

  @parameterized.named_parameters(
      ("mirrored", lambda d: MirroredStrategy(all_devices(d))),
      ("tpu_strategy", lambda d: maybe((d == "TPU"), TPUStrategy)))
  def test_check_tf_strategy(self, replicator_fn):
    replicator = replicator_fn(self.primary_device)
    if replicator is None:
      self.skipTest("Skipping test.")

    with self.assertRaisesRegexp(ValueError, "not compatible"):
      with replicator.scope():
        optimizer_utils.check_strategy()

  def test_check_updates_parameters_same_length(self):
    x = [42] * 5
    optimizer_utils.check_updates_parameters(updates=x, parameters=x)

  def test_check_updates_parameters_diff_length(self):
    updates = [42] * 2
    parameters = [42] * 3
    with self.assertRaisesRegexp(ValueError, "must be the same length"):
      optimizer_utils.check_updates_parameters(updates=updates,
                                               parameters=parameters)

  def test_check_updates_parameters_empty_params(self):
    x = []
    with self.assertRaisesRegexp(ValueError, "cannot be empty"):
      optimizer_utils.check_updates_parameters(updates=x, parameters=x)

  @parameterized.parameters(*itertools.product(DTYPES, DTYPES))
  def test_check_same_dtype(self, type_a, type_b):
    if self.primary_device == "TPU":
      if type_a == tf.float16:
        type_a = tf.bfloat16
      if type_b == tf.float16:
        type_b = tf.bfloat16

    a = tf.ones([], dtype=type_a)
    b = tf.ones([], dtype=type_b)

    if type_a == type_b:
      optimizer_utils.check_same_dtype(a, b)
    else:
      with self.assertRaisesRegexp(ValueError, "DType .* not equal"):
        optimizer_utils.check_same_dtype(a, b)

if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
