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
"""Tests for sonnet.v2.regularizers."""

import numpy as np
from sonnet.src import regularizers
from sonnet.src import test_utils
import tensorflow as tf


class L1Test(test_utils.TestCase):

  def testAgainstNumPy(self):
    regularizer = regularizers.L1(0.01)
    tensors = [tf.random.uniform([42]), tf.random.uniform([24])]

    def l1(scale, t):
      return scale * np.abs(t).sum()

    self.assertAllClose(
        regularizer(tensors),
        sum(l1(regularizer.scale, self.evaluate(t)) for t in tensors))

  def testNegativeScale(self):
    with self.assertRaises(ValueError):
      regularizers.L1(-1.0)

  def testEmpty(self):
    self.assertAllClose(regularizers.L1(0.01)([]), 0.0)


class L2Test(test_utils.TestCase):

  def testAgainstNumPy(self):
    regularizer = regularizers.L2(0.01)
    tensors = [tf.random.uniform([42]), tf.random.uniform([24])]

    def l2(scale, t):
      return scale * np.square(t).sum()

    self.assertAllClose(
        regularizer(tensors),
        sum(l2(regularizer.scale, self.evaluate(t)) for t in tensors))

  def testNegativeScale(self):
    with self.assertRaises(ValueError):
      regularizers.L2(-1.0)

  def testEmpty(self):
    self.assertAllClose(regularizers.L2(0.01)([]), 0.0)


class OffDiagonalOrthogonalTest(test_utils.TestCase):

  def testAgainstNumPy(self):
    regularizer = regularizers.OffDiagonalOrthogonal(0.01)
    tensors = [tf.random.uniform([4, 2]), tf.random.uniform([2, 4])]

    def odo(scale, t):
      t2 = np.square(np.dot(t.T, t))
      return scale * (t2.sum() - np.trace(t2))

    atol = 1e-3 if self.primary_device == "TPU" else 1e-6
    self.assertAllClose(
        regularizer(tensors),
        sum(odo(regularizer.scale, self.evaluate(t)) for t in tensors),
        atol=atol)

  def testNegativeScale(self):
    with self.assertRaises(ValueError):
      regularizers.OffDiagonalOrthogonal(-1.0)

  def testEmpty(self):
    self.assertAllClose(regularizers.OffDiagonalOrthogonal(0.01)([]), 0.0)


if __name__ == "__main__":
  tf.test.main()
