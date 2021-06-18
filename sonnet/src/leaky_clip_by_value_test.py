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
"""Tests for sonnet.v2.src.leaky_clip_by_value."""

from absl.testing import parameterized
from sonnet.src import leaky_clip_by_value
from sonnet.src import test_utils
import tensorflow as tf


class LeakyClipByValueTest(test_utils.TestCase, parameterized.TestCase):

  def test_leaky_clip_by_value_forward(self):
    t = tf.Variable([1.0, 2.0, 3.0])
    # Test when min/max are scalar values.
    clip_min = [1.5]
    clip_max = [2.5]
    clip_t = leaky_clip_by_value.leaky_clip_by_value(t, clip_min, clip_max)
    self.assertAllEqual(clip_t.numpy(), [1.5, 2.0, 2.5])
    # Test when min/max are of same sizes as t.
    clip_min_array = [0.5, 2.5, 2.5]
    clip_max_array = [1.5, 3.0, 3.5]
    clip_t_2 = leaky_clip_by_value.leaky_clip_by_value(t, clip_min_array,
                                                       clip_max_array)
    self.assertAllEqual(clip_t_2.numpy(), [1.0, 2.5, 3.0])

  @parameterized.parameters([
      (0.5, lambda x: x, [1.0]),
      (1.5, lambda x: x, [1.0]),
      (1.5, lambda x: -x, [0.0]),
      (-.5, lambda x: x, [0.0]),
      (-.5, lambda x: -x, [-1.0]),
  ])
  def test_leaky_clip_by_value_backward(self, init, fn, expected_grad):
    t = tf.Variable([init])
    max_val = 1.0
    min_val = 0.0
    with tf.GradientTape() as tape:
      clip_t = leaky_clip_by_value.leaky_clip_by_value(t, min_val, max_val)
      f = fn(clip_t)
    grad = tape.gradient(f, t)
    clip_t_value = clip_t.numpy()
    self.assertAllEqual(grad.numpy(), expected_grad)
    self.assertGreaterEqual(clip_t_value, min_val)
    self.assertLessEqual(clip_t_value, max_val)


if __name__ == "__main__":
  tf.test.main()
