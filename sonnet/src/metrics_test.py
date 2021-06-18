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
"""Tests for sonnet.v2.src.metrics."""

from sonnet.src import metrics
from sonnet.src import test_utils
import tensorflow as tf


class SumTest(test_utils.TestCase):

  def testSimple(self):
    acc = metrics.Sum()
    self.assertAllEqual([2., 3.], acc(tf.constant([2., 3.])))
    self.assertAllEqual([6., 8.], acc(tf.constant([4., 5.])))

  def testInitialize(self):
    acc = metrics.Sum()
    acc.initialize(tf.constant([1., 2.]))
    self.assertAllEqual([0., 0.], acc.value)

  def testReset(self):
    acc = metrics.Sum()
    self.assertAllEqual([2., 3.], acc(tf.constant([2., 3.])))
    self.assertAllEqual([6., 8.], acc(tf.constant([4., 5.])))
    acc.reset()
    self.assertAllEqual([7., 8.], acc(tf.constant([7., 8.])))


class MeanTest(test_utils.TestCase):

  def testSimple(self):
    mean = metrics.Mean()
    self.assertAllEqual([2., 3.], mean(tf.constant([2., 3.])))
    self.assertAllEqual([3., 4.], mean(tf.constant([4., 5.])))

  def testInitialize(self):
    mean = metrics.Mean()
    mean.initialize(tf.constant([1., 2.]))
    self.assertAllEqual([1., 2.], mean(tf.constant([1., 2.])))

  def testReset(self):
    mean = metrics.Mean()
    self.assertAllEqual([2., 3.], mean(tf.constant([2., 3.])))
    self.assertAllEqual([3., 4.], mean(tf.constant([4., 5.])))
    mean.reset()
    self.assertAllEqual([7., 8.], mean(tf.constant([7., 8.])))


if __name__ == "__main__":
  tf.test.main()
