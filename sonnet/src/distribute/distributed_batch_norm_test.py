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
"""Tests for sonnet.v2.src.distribute.batch_norm."""

from absl import logging

from absl.testing import parameterized
from sonnet.src import test_utils
from sonnet.src.distribute import distributed_batch_norm as batch_norm
from sonnet.src.distribute import replicator
import tensorflow as tf


class CrossReplicaBatchNormTest(test_utils.TestCase, parameterized.TestCase):
  # Avoid running tests inside a `with tf.device("TPU:0"):` block.
  ENTER_PRIMARY_DEVICE = False

  def testDefaultReplicaContext(self):
    layer = batch_norm.CrossReplicaBatchNorm(False, False, TestMetric(),
                                             TestMetric())

    inputs = tf.ones([2, 3, 3, 5])
    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    outputs = layer(inputs, True, scale=scale, offset=offset).numpy()
    self.assertAllEqual(outputs, tf.fill(inputs.shape, 2.0))

  def testWithMultipleDevicesMirrored(self):
    if self.primary_device == "CPU":
      self.skipTest("No devices to mirror across.")
    elif self.primary_device == "GPU":
      devices = tf.config.experimental.list_logical_devices("GPU")
    else:
      devices = tf.config.experimental.list_logical_devices("TPU")

    strategy = replicator.Replicator([device.name for device in devices])
    with strategy.scope():
      mean_metric = TestMetric()
      var_metric = TestMetric()
      layer = batch_norm.CrossReplicaBatchNorm(False, False, mean_metric,
                                               var_metric)

    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    def foo():
      inputs = tf.random.normal([2, 3, 3, 5])
      outputs = layer(inputs, True, False, scale, offset)
      return inputs, outputs

    inputs, outputs = strategy.run(foo)
    local_mean_metric = strategy.experimental_local_results(mean_metric.value)
    local_var_metric = strategy.experimental_local_results(var_metric.value)
    self.assertAllEqual(local_mean_metric[0].numpy(),
                        local_mean_metric[1].numpy())
    self.assertAllEqual(local_var_metric[0].numpy(),
                        local_var_metric[1].numpy())
    mean = local_mean_metric[0]
    var = local_var_metric[0]

    for inp, out in zip(
        strategy.experimental_local_results(inputs),
        strategy.experimental_local_results(outputs)):
      expected_out = (inp - mean) * tf.math.rsqrt(var + 1e-5) * scale + offset
      self.assertAllClose(out, expected_out)

  def testWithTpuStrategy(self):
    if self.primary_device != "TPU":
      self.skipTest("TPU strategy only runs on TPU's")

    strategy = replicator.TpuReplicator()
    with strategy.scope():
      mean_metric = TestMetric()
      var_metric = TestMetric()
      layer = batch_norm.CrossReplicaBatchNorm(False, False,
                                               mean_metric, var_metric)
    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    @tf.function
    def run():
      def compute():
        inputs = tf.ones([2, 3, 3, 5])
        outputs = layer(inputs, True, False, scale, offset)
        return inputs, outputs

      return strategy.run(compute)
    inputs, outputs = run()

    local_mean_metric = strategy.experimental_local_results(mean_metric.value)
    local_var_metric = strategy.experimental_local_results(var_metric.value)
    self.assertAllEqual(local_mean_metric[0].numpy(),
                        local_mean_metric[1].numpy())
    self.assertAllEqual(local_var_metric[0].numpy(),
                        local_var_metric[1].numpy())
    mean = local_mean_metric[0]
    var = local_var_metric[0]

    for inp, out in zip(
        strategy.experimental_local_results(inputs),
        strategy.experimental_local_results(outputs)):
      expected_out = (inp - mean) * tf.math.rsqrt(var + 1e-5) * scale + offset
      self.assertAllClose(out, expected_out)


class TestMetric:

  def __init__(self):
    self._foo = None
    self._built = False

  def update(self, x):
    if self._built:
      self._foo.assign(x)
    else:
      self._foo = tf.Variable(x)
      self._built = True

  @property
  def value(self):
    return self._foo

  def initialize(self, x):
    self._foo = tf.Variable(x)
    self._built = True


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
