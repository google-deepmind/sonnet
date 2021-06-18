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
"""Tests for sonnet.v2.src.moving_averages."""

from absl.testing import parameterized
from sonnet.src import moving_averages
from sonnet.src import test_utils
import tensorflow as tf


class ExponentialMovingAverageTest(test_utils.TestCase, parameterized.TestCase):

  def testCall(self):
    ema = moving_averages.ExponentialMovingAverage(0.50)

    self.assertAllClose(ema(3.0).numpy(), 3.0)
    self.assertAllClose(ema(6.0).numpy(), 5.0)

  def testUpdateAndValue(self):
    ema = moving_averages.ExponentialMovingAverage(0.50)
    ema.update(3.0)
    self.assertAllClose(ema.value.numpy(), 3.0, atol=1e-3, rtol=1e-5)

    ema.update(6.0)
    self.assertAllClose(ema.value.numpy(), 5.0, atol=1e-3, rtol=1e-5)

  def testReset(self):
    ema = moving_averages.ExponentialMovingAverage(0.90)
    self.assertAllClose(ema(3.0).numpy(), 3.0, atol=1e-3, rtol=1e-5)

    ema.reset()
    self.assertEqual(ema.value.shape, ())
    self.assertEqual(ema.value.numpy(), 0.0)

    self.assertAllClose(ema(3.0).numpy(), 3.0, atol=1e-3, rtol=1e-5)

  def testResetVector(self):
    ema = moving_averages.ExponentialMovingAverage(0.90)
    random_input = tf.random.normal((1, 5))
    ema(random_input)
    ema.reset()
    self.assertEqual(ema.value.shape, (1, 5))
    self.assertAllClose(ema.value.numpy(), tf.zeros_like(random_input))
    self.assertEqual(ema._counter.dtype, tf.int64)

  def testValueEqualsLatestUpdate(self):
    ema = moving_averages.ExponentialMovingAverage(0.50)

    self.assertAllClose(ema(3.0).numpy(), 3.0, atol=1e-3, rtol=1e-5)
    self.assertAllClose(ema.value.numpy(), 3.0, atol=1e-3, rtol=1e-5)

    self.assertAllClose(ema(6.0).numpy(), 5.0, atol=1e-3, rtol=1e-5)
    self.assertAllClose(ema.value.numpy(), 5.0, atol=1e-3, rtol=1e-5)

  @parameterized.parameters(True, False)
  def testWithTFFunction(self, autograph):
    ema_1 = moving_averages.ExponentialMovingAverage(0.95)
    ema_2 = moving_averages.ExponentialMovingAverage(0.95)
    ema_func = tf.function(ema_2, autograph=autograph)

    for _ in range(10):
      x = tf.random.uniform((), 0, 10)
      self.assertAllClose(
          ema_1(x).numpy(), ema_func(x).numpy(), atol=1e-3, rtol=1e-5)

  @parameterized.parameters(True, False)
  def testResetWithTFFunction(self, autograph):
    ema = moving_averages.ExponentialMovingAverage(0.90)
    ema_func = tf.function(ema, autograph=autograph)
    self.assertAllClose(ema_func(3.0).numpy(), 3.0, atol=1e-3, rtol=1e-5)

    ema.reset()
    self.assertEqual(ema.value.numpy(), 0.0)

    self.assertAllClose(ema_func(3.0).numpy(), 3.0, atol=1e-3, rtol=1e-5)

  @parameterized.named_parameters(("2D", [2, 2]), ("3D", [1, 1, 3]))
  def testAlternativeShape(self, shape):
    ema = moving_averages.ExponentialMovingAverage(0.90)
    value = tf.random.uniform(shape)
    result = ema(value)
    self.assertEqual(value.shape, result.shape)


if __name__ == "__main__":
  tf.test.main()
