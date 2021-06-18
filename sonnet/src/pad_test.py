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
"""Tests for sonnet.v2.src.pad."""

from absl.testing import parameterized
from sonnet.src import pad
from sonnet.src import test_utils
import tensorflow as tf


class PadTest(test_utils.TestCase, parameterized.TestCase):

  def test_padding_2d(self):
    a = pad.create([pad.causal, pad.full], [3], [1, 1], 2, -1)
    self.assertEqual(a, [[0, 0], [2, 0], [2, 2], [0, 0]])

  def test_padding_1d(self):
    a = pad.create(pad.full, 3, 1, 1, 1)
    self.assertEqual(a, [[0, 0], [0, 0], [2, 2]])

  def test_padding_3d(self):
    a = pad.create([pad.causal, pad.full, pad.full], [3, 2, 3], [1], 3, -1)
    self.assertEqual(a, [[0, 0], [2, 0], [1, 1], [2, 2], [0, 0]])

  @parameterized.parameters((2, [2, 2]), (3, [4, 4, 4, 4]), ([2, 2], 3),
                            ([4, 4, 4, 4], 3))
  def test_padding_incorrect_input(self, kernel_size, rate):
    with self.assertRaisesRegex(
        TypeError,
        r"must be a scalar or sequence of length 1 or sequence of length 3."):
      pad.create(pad.full, kernel_size, rate, 3, -1)

  def test_padding_valid(self):
    a = pad.create(pad.valid, 4, 3, 2, -1)
    self.assertEqual(a, [[0, 0], [0, 0], [0, 0], [0, 0]])

  def test_padding_same(self):
    a = pad.create(pad.same, 4, 3, 2, -1)
    self.assertEqual(a, [[0, 0], [4, 5], [4, 5], [0, 0]])

  def test_padding_full(self):
    a = pad.create(pad.full, 4, 3, 2, -1)
    self.assertEqual(a, [[0, 0], [9, 9], [9, 9], [0, 0]])

  def test_padding_causal(self):
    a = pad.create(pad.causal, 4, 3, 2, -1)
    self.assertEqual(a, [[0, 0], [9, 0], [9, 0], [0, 0]])

  def test_padding_reverse_causal(self):
    a = pad.create(pad.reverse_causal, 4, 3, 2, -1)
    self.assertEqual(a, [[0, 0], [0, 9], [0, 9], [0, 0]])

  @parameterized.parameters((1, 1, 1), (3, 1, 1), (1, 3, 1), (1, 1, 3),
                            (3, 3, 1), (3, 1, 3), (1, 3, 3), (3, 3, 3))
  def test_same_padding(self, kernel_size, stride, rate):
    a = tf.random.normal([2, 4, 3])
    k = tf.random.normal([kernel_size, 3, 4])
    padding = pad.create(pad.same, kernel_size, rate, 1, -1)
    a_padded = tf.pad(a, padding)
    y1 = tf.nn.conv1d(
        a_padded, k, stride=stride, dilations=rate, padding="VALID")
    y2 = tf.nn.conv1d(a, k, stride=stride, dilations=rate, padding="SAME")
    self.assertEqual(y1.shape, y2.shape)
    self.assertAllClose(y1.numpy(), y2.numpy())

  @parameterized.parameters((1, 1, 1), (3, 1, 1), (1, 3, 1), (1, 1, 3),
                            (3, 3, 1), (3, 1, 3), (1, 3, 3), (3, 3, 3))
  def test_valid_padding(self, kernel_size, stride, rate):
    a = tf.random.normal([2, 8, 3])
    k = tf.random.normal([kernel_size, 3, 4])
    padding = pad.create(pad.valid, kernel_size, rate, 1, -1)
    a_padded = tf.pad(a, padding)
    y1 = tf.nn.conv1d(
        a_padded, k, stride=stride, dilations=rate, padding="VALID")
    y2 = tf.nn.conv1d(a, k, stride=stride, dilations=rate, padding="VALID")
    self.assertAllEqual(y1.numpy(), y2.numpy())


if __name__ == "__main__":
  tf.test.main()
