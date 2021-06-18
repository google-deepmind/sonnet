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
"""Tests for sonnet.v2.src.nets.dnc.write."""

import numpy as np
from sonnet.src import test_utils
from sonnet.src.nets.dnc import write
import tensorflow as tf


class EraseRowsTest(test_utils.TestCase):

  def testShape(self):
    batch_size = 16
    num_writes = 2
    memory_size = 5
    word_size = 3

    mem = tf.random.uniform([batch_size, memory_size, word_size])
    write_address = tf.random.uniform([batch_size, num_writes, memory_size])
    reset_row_weights = tf.random.uniform([batch_size, num_writes])
    eraser = write.erase_rows(mem, write_address, reset_row_weights)
    self.assertAllEqual(eraser.shape.as_list(),
                        [batch_size, memory_size, word_size])

  def testValues(self):
    num_writes = 2
    memory_size = 5
    word_size = 3

    # Random memory, weights and values (batch_size=1)
    mem = tf.random.uniform((1, memory_size, word_size))
    mem_np = mem.numpy()
    # Non-repeated indices in [0, memory_size)
    perm = np.random.permutation(memory_size)
    indices_np = perm[:num_writes]
    excluded_indices_np = perm[num_writes:]

    # One-hot representation
    write_address = tf.constant(
        np.expand_dims(np.eye(memory_size)[indices_np], axis=0),
        dtype=tf.float32)
    reset_row_weights = tf.ones((1, num_writes))

    erased_mem = write.erase_rows(mem, write_address, reset_row_weights)

    not_erased_mem = write.erase_rows(mem, write_address, reset_row_weights * 0)

    erased_mem_np = erased_mem.numpy()

    # Rows specified in indices should have been erased.
    self.assertAllClose(
        erased_mem_np[0, indices_np, :],
        np.zeros((num_writes, word_size)),
        atol=2e-3)

    # Other rows should not have been erased.
    self.assertAllClose(
        erased_mem_np[0, excluded_indices_np, :],
        mem_np[0, excluded_indices_np, :],
        atol=2e-3)

    # Write with reset weights zero'd out and nothing should change.
    self.assertAllEqual(not_erased_mem.numpy(), mem_np)


class EraseTest(test_utils.TestCase):

  def testShape(self):
    batch_size = 1
    num_writes = 2
    memory_size = 5
    word_size = 3

    mem = tf.random.uniform([batch_size, memory_size, word_size])
    write_address = tf.random.uniform([batch_size, num_writes, memory_size])
    reset_weights = tf.random.uniform([batch_size, num_writes, word_size])
    writer = write.erase(mem, write_address, reset_weights)
    self.assertTrue(writer.shape.as_list(),
                    [batch_size, memory_size, word_size])

  def testValues(self):
    num_writes = 2
    memory_size = 5
    word_size = 3

    # Random memory, weights and values (batch_size=1)
    mem = tf.random.uniform([1, memory_size, word_size])
    mem_np = mem.numpy()
    # Non-repeated indices in [0, memory_size)
    perm = np.random.permutation(memory_size)
    indices = perm[:num_writes]
    excluded_indices = perm[num_writes:]
    # One-hot representation
    write_address = tf.constant(
        np.expand_dims(np.eye(memory_size)[indices], axis=0), dtype=tf.float32)
    reset_weights = tf.ones([1, num_writes, word_size])
    erased_mem = write.erase(mem, write_address, reset_weights)
    not_erased_mem = write.erase(mem, write_address, reset_weights * 0.)

    erased_mem_np = erased_mem.numpy()
    not_erased_mem_np = not_erased_mem.numpy()

    # Rows specified in indices should have been erased.
    self.assertAllClose(
        erased_mem_np[0, indices, :],
        np.zeros((num_writes, word_size)),
        atol=2e-3)

    # Other rows should not have been erased.
    self.assertAllClose(
        erased_mem_np[0, excluded_indices, :],
        mem_np[0, excluded_indices, :],
        atol=2e-3)

    # Write with reset weights zero'd out and nothing should change.
    self.assertAllEqual(not_erased_mem_np, mem_np)


class EraseAndWriteTest(test_utils.TestCase):

  def testShape(self):
    batch_size = 4
    num_writes = 2
    memory_size = 5
    word_size = 3

    mem = tf.random.uniform([batch_size, memory_size, word_size])
    write_address = tf.random.uniform([batch_size, num_writes, memory_size])
    reset_weights = tf.random.uniform([batch_size, num_writes, word_size])
    values = tf.random.uniform([batch_size, num_writes, word_size])
    writer = write.erase_and_write(mem, write_address, reset_weights, values)
    self.assertTrue(writer.shape.as_list(),
                    [batch_size, memory_size, word_size])

  def testValues(self):
    batch_size = 4
    num_writes = 2
    memory_size = 5
    word_size = 3

    # Random memory, weights and values (batch_size=1)
    mem = tf.random.uniform((batch_size, memory_size, word_size))
    # Non-repeated indices in [0, memory_size)
    indices = np.random.permutation(memory_size)[:num_writes]
    # One-hot representation
    write_address = tf.constant(
        np.tile(np.eye(memory_size)[indices], [batch_size, 1, 1]),
        dtype=tf.float32)

    reset_weights = tf.ones((batch_size, num_writes, word_size), 1)
    write_values = tf.random.uniform([batch_size, num_writes, word_size])

    written_mem = write.erase_and_write(mem, write_address, reset_weights,
                                        write_values)

    self.assertAllClose(
        written_mem.numpy()[0, indices, :], write_values.numpy()[0], atol=2e-3)


class AdditiveWriteTest(test_utils.TestCase):

  def testShape(self):
    batch_size = 4
    num_writes = 2
    memory_size = 5
    word_size = 3

    mem = tf.random.uniform([batch_size, memory_size, word_size])
    write_address = tf.random.uniform([batch_size, num_writes, memory_size])
    values = tf.random.uniform([batch_size, num_writes, word_size])
    writer = write.additive_write(mem, write_address, values)
    self.assertAllEqual(writer.shape.as_list(),
                        [batch_size, memory_size, word_size])

  def testValues(self):
    num_writes = 2
    memory_size = 5
    word_size = 3

    # Random memory, address and values (batch_size=1)
    mem = tf.random.uniform([1, memory_size, word_size])
    mem_np = mem.numpy()
    # Non-repeated indices in [0, memory_size)
    indices = np.random.permutation(memory_size)[:num_writes]
    # One-hot representation
    write_address = tf.constant(
        np.expand_dims(np.eye(memory_size)[indices], axis=0), dtype=tf.float32)
    write_values = tf.random.uniform([1, num_writes, word_size])
    write_values_np = write_values.numpy()

    written_mem = write.additive_write(mem, write_address, write_values)
    not_written_mem = write.additive_write(mem, write_address * 0, write_values)

    written_mem_np = written_mem.numpy()
    not_written_mem_np = not_written_mem.numpy()

    # Check values have been correctly written
    self.assertAllClose(
        written_mem.numpy()[0, indices, :],
        write_values_np[0] + mem_np[0, indices, :],
        atol=2e-3)
    # Check all other values in the memory are still what they started as
    written_mem_copy = written_mem_np.copy()
    written_mem_copy[0, indices, :] -= write_values_np[0]
    self.assertAllClose(written_mem_copy, mem_np, atol=2e-3)

    self.assertAllClose(not_written_mem_np, mem_np, atol=2e-3)


if __name__ == '__main__':
  tf.test.main()
