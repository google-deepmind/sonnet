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
"""Tests for sonnet.v2.examples.simple_mnist."""

import sonnet as snt
from examples import simple_mnist
from sonnet.src import test_utils
import tensorflow as tf


class SimpleMnistTest(test_utils.TestCase):

  def setUp(self):
    self.ENTER_PRIMARY_DEVICE = False  # pylint: disable=invalid-name
    super().setUp()

  def test_train_epoch(self):
    model = snt.Sequential([
        snt.Flatten(),
        snt.Linear(10),
    ])

    optimizer = snt.optimizers.SGD(0.1)

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([2, 8, 8, 1]),
         tf.ones([2], dtype=tf.int64))).batch(2).repeat(4)

    for _ in range(3):
      loss = simple_mnist.train_epoch(model, optimizer, dataset)
    self.assertEqual(loss.shape, [])
    self.assertEqual(loss.dtype, tf.float32)

  def test_test_accuracy(self):
    model = snt.Sequential([
        snt.Flatten(),
        snt.Linear(10),
    ])
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([2, 8, 8, 1]),
         tf.ones([2], dtype=tf.int64))).batch(2).repeat(4)

    outputs = simple_mnist.test_accuracy(model, dataset)
    self.assertEqual(len(outputs), 2)


if __name__ == "__main__":
  tf.test.main()
