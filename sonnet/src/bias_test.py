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
"""Tests for sonnet.v2.src.bias."""

from sonnet.src import bias
from sonnet.src import test_utils
import tensorflow as tf


class BiasTest(test_utils.TestCase):

  def test_output_shape(self):
    mod = bias.Bias(output_size=(2 * 2,))
    with self.assertRaisesRegex(ValueError, "Input shape must be [(]-1, 4[)]"):
      mod(tf.ones([2, 2, 2]))

  def test_output_size_valid(self):
    mod = bias.Bias(output_size=(2 * 2,))
    mod(tf.ones([2, 2 * 2]))

  def test_bias_dims_scalar(self):
    mod = bias.Bias(bias_dims=())
    mod(tf.ones([1, 2, 3, 4]))
    self.assertEmpty(mod.b.shape)

  def test_bias_dims_custom(self):
    b, d1, d2, d3 = range(1, 5)
    mod = bias.Bias(bias_dims=[1, 3])
    out = mod(tf.ones([b, d1, d2, d3]))
    self.assertEqual(mod.b.shape, [d1, 1, d3])
    self.assertEqual(out.shape, [b, d1, d2, d3])

  def test_bias_dims_negative_out_of_order(self):
    mod = bias.Bias(bias_dims=[-1, -2])
    mod(tf.ones([1, 2, 3]))
    self.assertEqual(mod.b.shape, [2, 3])

  def test_bias_dims_invalid(self):
    mod = bias.Bias(bias_dims=[1, 5])
    with self.assertRaisesRegex(ValueError,
                                "5 .* out of range for input of rank 3"):
      mod(tf.ones([1, 2, 3]))

  def test_b_init_defaults_to_zeros(self):
    mod = bias.Bias()
    mod(tf.ones([1, 1]))
    self.assertAllEqual(mod.b.read_value(), tf.zeros_like(mod.b))

  def test_b_init_custom(self):
    ones_initializer = lambda s, d: tf.ones(s, dtype=d)
    mod = bias.Bias(b_init=ones_initializer)
    mod(tf.ones([1, 1]))
    self.assertAllEqual(mod.b.read_value(), tf.ones_like(mod.b))

  def test_name(self):
    mod = bias.Bias(name="foo")
    self.assertEqual(mod.name, "foo")
    mod(tf.ones([1, 1]))
    self.assertEqual(mod.b.name, "foo/b:0")

  def test_multiplier(self):
    ones_initializer = lambda s, d: tf.ones(s, dtype=d)
    mod = bias.Bias(b_init=ones_initializer)
    out = mod(tf.ones([1, 1]), multiplier=-1)
    self.assertAllEqual(tf.reduce_sum(out), 0)


if __name__ == "__main__":
  tf.test.main()
