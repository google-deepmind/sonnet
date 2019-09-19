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
"""Tests for sonnet.v2.src.batch_norm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import logging

from absl.testing import parameterized
import numpy as np
from sonnet.src import batch_norm
from sonnet.src import initializers
from sonnet.src import replicator
from sonnet.src import test_utils
import tensorflow as tf


class BaseBatchNormTest(test_utils.TestCase, parameterized.TestCase):

  def testSimpleTraining(self):
    layer = batch_norm.BaseBatchNorm(
        moving_mean=TestMetric(),
        moving_variance=TestMetric(),
        create_scale=False,
        create_offset=False)

    inputs = tf.ones([2, 3, 3, 5])
    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    outputs = layer(inputs, True, scale=scale, offset=offset).numpy()
    self.assertAllEqual(outputs, tf.fill(inputs.shape, 2.0))
    self.assertEqual((0, 1, 2), layer._axis)

  def testSimpleTrainingNCHW(self):
    layer = batch_norm.BaseBatchNorm(
        moving_mean=TestMetric(),
        moving_variance=TestMetric(),
        create_scale=False,
        create_offset=False,
        data_format="NCHW")

    inputs = tf.ones([2, 5, 3, 3])
    scale = tf.constant(0.5, shape=(5, 1, 1))
    offset = tf.constant(2.0, shape=(5, 1, 1))

    outputs = layer(inputs, True, scale=scale, offset=offset).numpy()
    self.assertAllEqual(outputs, tf.fill(inputs.shape, 2.0))
    self.assertEqual((0, 2, 3), layer._axis)

  def testSimpleTraining3D(self):
    layer = batch_norm.BaseBatchNorm(
        moving_mean=TestMetric(),
        moving_variance=TestMetric(),
        create_scale=False,
        create_offset=False)

    inputs = tf.ones([2, 3, 3, 3, 5])
    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    outputs = layer(inputs, True, scale=scale, offset=offset).numpy()
    self.assertAllEqual(outputs, tf.fill(inputs.shape, 2.0))
    self.assertEqual((0, 1, 2, 3), layer._axis)

  def testSimpleTraining3DNCDHW(self):
    layer = batch_norm.BaseBatchNorm(
        moving_mean=TestMetric(),
        moving_variance=TestMetric(),
        create_scale=False,
        create_offset=False,
        data_format="NCDHW")

    inputs = tf.ones([2, 5, 3, 3, 3])
    scale = tf.constant(0.5, shape=(5, 1, 1, 1))
    offset = tf.constant(2.0, shape=(5, 1, 1, 1))

    outputs = layer(inputs, True, scale=scale, offset=offset).numpy()
    self.assertAllEqual(outputs, tf.fill(inputs.shape, 2.0))
    self.assertEqual((0, 2, 3, 4), layer._axis)

  def testNoScaleAndOffset(self):
    layer = batch_norm.BaseBatchNorm(
        moving_mean=TestMetric(),
        moving_variance=TestMetric(),
        create_scale=False,
        create_offset=False,
        data_format="NHWC")

    inputs = tf.ones([2, 5, 3, 3, 3])
    outputs = layer(inputs, True)
    self.assertAllEqual(outputs, tf.zeros_like(inputs))

  @parameterized.parameters(True, False)
  def testWithTfFunction(self, autograph):
    if "TPU" in self.device_types:
      self.skipTest("Test not working on TPU")
      # TODO(tamaranorman) enable on TPU

    layer = batch_norm.BaseBatchNorm(
        moving_mean=TestMetric(),
        moving_variance=TestMetric(),
        create_scale=False,
        create_offset=False,
        data_format="NHWC")
    layer = tf.function(layer, autograph=autograph)

    inputs = tf.ones([2, 5, 3, 3, 3])
    scale = tf.constant(0.5, shape=(5, 1, 1, 1))
    offset = tf.constant(2.0, shape=(5, 1, 1, 1))
    expected1 = tf.zeros_like(inputs)
    expected2 = tf.fill(inputs.shape, 2.0)

    for is_training, use_batch_stats in itertools.product((True, False),
                                                          (True, False)):
      outputs = layer(inputs, is_training, use_batch_stats)
      self.assertAllEqual(outputs, expected1)

      outputs = layer(
          inputs, is_training, use_batch_stats, scale=scale, offset=offset)
      self.assertAllEqual(outputs, expected2)

  @parameterized.parameters(True, False)
  def testWithTfFunctionTfArgs(self, autograph):
    if "TPU" in self.device_types:
      self.skipTest("Test not working on TPU")
      # TODO(tamaranorman) enable on TPU

    layer = batch_norm.BaseBatchNorm(
        moving_mean=TestMetric(),
        moving_variance=TestMetric(),
        create_scale=False,
        create_offset=False,
        data_format="NHWC")
    layer = tf.function(layer, autograph=autograph)

    inputs = tf.ones([2, 5, 3, 3, 3])
    expected = tf.zeros_like(inputs)

    for is_training, use_batch_stats in itertools.product((True, False),
                                                          (True, False)):
      # NOTE: The use of `tf.constant` means we require graph control flow
      outputs = layer(inputs, tf.constant(is_training),
                      tf.constant(use_batch_stats))
      self.assertAllEqual(outputs, expected)

  def testUsingTestStats(self):
    layer = batch_norm.BaseBatchNorm(
        moving_mean=TestMetric(),
        moving_variance=TestMetric(),
        create_scale=False,
        create_offset=False)

    inputs = tf.ones([2, 3, 3, 5])
    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    outputs = layer(inputs, True, scale=scale, offset=offset).numpy()
    self.assertAllEqual(outputs, tf.fill(inputs.shape, 2.0))
    outputs = layer(inputs, False, scale=scale, offset=offset).numpy()
    for x in np.nditer(outputs):
      self.assertAllClose(x, 2.0, rtol=1e-5, atol=1e-3)

  def testIsTrainingFalseFirstCall(self):
    layer = batch_norm.BaseBatchNorm(
        moving_mean=TestMetric(),
        moving_variance=TestMetric(),
        create_scale=False,
        create_offset=False)
    inputs = tf.ones([2, 3, 3, 5])
    outputs = layer(inputs, False)
    self.assertAllEqual(outputs, tf.fill(inputs.shape, 0.0))

  @parameterized.parameters("NHW", "HWC", "channel_last")
  def testInvalidDataFormat(self, data_format):
    with self.assertRaisesRegexp(
        ValueError,
        "Unable to extract channel information from '{}'".format(data_format)):
      batch_norm.BaseBatchNorm(
          moving_mean=TestMetric(),
          moving_variance=TestMetric(),
          create_scale=False,
          create_offset=False,
          data_format=data_format)

  @parameterized.parameters("NCHW", "NCW", "channels_first")
  def testValidDataFormatChannelsFirst(self, data_format):
    test = batch_norm.BaseBatchNorm(
        moving_mean=TestMetric(),
        moving_variance=TestMetric(),
        create_scale=False,
        create_offset=False,
        data_format=data_format)

    self.assertEqual(test._channel_index, 1)

  @parameterized.parameters("NHWC", "NWC", "channels_last")
  def testValidDataFormatChannelsLast(self, data_format):
    test = batch_norm.BaseBatchNorm(
        moving_mean=TestMetric(),
        moving_variance=TestMetric(),
        create_scale=False,
        create_offset=False,
        data_format=data_format)

    self.assertEqual(test._channel_index, -1)

  def testNoScaleAndInitProvided(self):
    with self.assertRaisesRegexp(
        ValueError, "Cannot set `scale_init` if `create_scale=False`"):
      batch_norm.BaseBatchNorm(
          moving_mean=TestMetric(),
          moving_variance=TestMetric(),
          create_scale=False,
          create_offset=True,
          scale_init=initializers.Ones())

  def testNoOffsetBetaInitProvided(self):
    with self.assertRaisesRegexp(
        ValueError, "Cannot set `offset_init` if `create_offset=False`"):
      batch_norm.BaseBatchNorm(
          moving_mean=TestMetric(),
          moving_variance=TestMetric(),
          create_scale=True,
          create_offset=False,
          offset_init=initializers.Zeros())


class BatchNormTest(test_utils.TestCase, parameterized.TestCase):

  def testSimple(self):
    layer = batch_norm.BatchNorm(False, False)

    inputs = tf.ones([2, 3, 3, 5])
    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    outputs = layer(inputs, True, scale=scale, offset=offset).numpy()
    self.assertAllEqual(outputs, tf.fill(inputs.shape, 2.0))


class CrossReplicaBatchNormTest(test_utils.TestCase, parameterized.TestCase):
  # TODO(tamaranorman) add a TpuReplicator test

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

    inputs, outputs = strategy.experimental_run_v2(foo)
    self.assertAllEqual(mean_metric.value.values[0].numpy(),
                        mean_metric.value.values[1].numpy())
    self.assertAllEqual(var_metric.value.values[0].numpy(),
                        var_metric.value.values[1].numpy())
    mean = mean_metric.value.values[0]
    var = var_metric.value.values[0]

    for inp, out in zip(inputs.values, outputs.values):
      expected_out = (inp - mean) * tf.math.rsqrt(var + 1e-5) * scale + offset
      self.assertAllClose(out, expected_out)


class TestMetric(object):

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
  # tf.enable_v2_behavior()
  tf.test.main()
