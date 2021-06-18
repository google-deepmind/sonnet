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
"""Tests for sonnet.v2.src.nets.dnc.read."""

import numpy as np
from sonnet.src import test_utils
from sonnet.src.nets.dnc import read
import tensorflow as tf


class ReadTest(test_utils.TestCase):

  def testShape(self):
    batch_size = 4
    num_reads = 2
    memory_size = 5
    word_size = 3

    mem = tf.random.uniform([batch_size, memory_size, word_size])
    weights = tf.random.uniform([batch_size, num_reads, memory_size])
    values_read = read.read(mem, weights)
    self.assertAllEqual(values_read.shape.as_list(),
                        [batch_size, num_reads, word_size])

  def testValues(self):
    num_reads = 2
    memory_size = 5
    word_size = 3

    # Random memory and weights (batch_size=1)
    mem = tf.random.uniform([1, memory_size, word_size])
    indices = np.random.randint(0, memory_size, size=num_reads)
    # One-hot representation
    read_weights = tf.constant(
        np.expand_dims(np.eye(memory_size)[indices], axis=0), dtype=tf.float32)

    read_values = read.read(mem, read_weights, squash_op=tf.identity)
    self.assertAllClose(
        mem.numpy()[0, indices, :], read_values.numpy()[0, ...], atol=2e-3)


if __name__ == '__main__':
  tf.test.main()
