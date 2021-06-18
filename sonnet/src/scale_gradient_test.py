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
"""Tests for sonnet.v2.src.scale_gradient."""

import itertools

from absl.testing import parameterized
from sonnet.src import scale_gradient
from sonnet.src import test_utils
import tensorflow as tf


class ScaleGradientTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      *itertools.product([-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5, 2.0]))
  def test_scale(self, t_, scale):
    t = tf.Variable([t_])
    with tf.GradientTape() as tape:
      y = scale_gradient.scale_gradient(t, scale)
      output = y * y
    grad = tape.gradient(output, t)
    self.assertAllEqual(grad.numpy(), [2 * t_ * scale])
    self.assertAllEqual(output.numpy(), [t_**2])


if __name__ == "__main__":
  tf.test.main()
