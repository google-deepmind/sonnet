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
"""Tests for sonnet.v2.src.parallel_linear."""

from sonnet.src import linear
from sonnet.src import parallel_linear
from sonnet.src import test_utils
import tensorflow as tf


class ParallelLinearTest(test_utils.TestCase):

  def test_output_size_correct(self):
    layer = parallel_linear.ParallelLinears(3)

    outputs = layer(tf.ones([4, 2, 6]))
    self.assertEqual(outputs.shape, [4, 2, 3])

  def test_behaves_same_as_stacked_linears(self):
    w_init = tf.random.normal((3, 5, 7))
    b_init = tf.random.normal((3, 1, 7))
    inputs = tf.random.normal((3, 2, 5))

    parallel = parallel_linear.ParallelLinears(
        7, w_init=lambda s, d: w_init, b_init=lambda s, d: b_init)
    parallel_outputs = parallel(inputs)

    stacked_outputs = []
    for i in range(3):
      layer = linear.Linear(
          7,
          w_init=lambda s, d, i=i: w_init[i],
          b_init=lambda s, d, i=i: b_init[i])
      stacked_outputs.append(layer(inputs[i]))
    stacked_outputs = tf.stack(stacked_outputs, axis=0)

    self.assertAllClose(parallel_outputs.numpy(), stacked_outputs.numpy())


if __name__ == '__main__':
  tf.test.main()
