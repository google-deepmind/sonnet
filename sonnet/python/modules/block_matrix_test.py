# Copyright 2017 The Sonnet Authors. All Rights Reserved.
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

"""Tests for block_matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from sonnet.python.modules import block_matrix
import tensorflow as tf


def create_input(size, batch_size=1):
  x = tf.range(size * batch_size)
  return tf.reshape(tf.to_float(x), shape=(batch_size, -1))


class BlockTriangularMatrixTest(tf.test.TestCase):

  def _check_output_size(self, btm, result, batch_size=1):
    self.assertEqual(result.shape, (batch_size,) + btm.output_shape)

  def test_lower(self):
    """Tests block lower-triangular matrix."""

    btm = block_matrix.BlockTriangularMatrix(
        block_shape=(2, 3), block_rows=3, upper=False)
    self.assertEqual(btm.num_blocks, 6)
    self.assertEqual(btm.block_size, 6)
    self.assertEqual(btm.input_size, 36)

    output = btm(create_input(btm.input_size))
    with self.test_session() as sess:
      result = sess.run(output)

    self._check_output_size(btm, result)

    expected = np.array([[[0, 1, 2, 0, 0, 0, 0, 0, 0],
                          [3, 4, 5, 0, 0, 0, 0, 0, 0],
                          [6, 7, 8, 9, 10, 11, 0, 0, 0],
                          [12, 13, 14, 15, 16, 17, 0, 0, 0],
                          [18, 19, 20, 21, 22, 23, 24, 25, 26],
                          [27, 28, 29, 30, 31, 32, 33, 34, 35]]])
    self.assertAllEqual(result, expected)

  def test_lower_no_diagonal(self):
    """Tests block lower-triangular matrix without diagonal."""

    btm = block_matrix.BlockTriangularMatrix(
        block_shape=(2, 3), block_rows=3, include_diagonal=False)
    self.assertEqual(btm.num_blocks, 3)
    self.assertEqual(btm.block_size, 6)
    self.assertEqual(btm.input_size, 18)

    output = btm(create_input(btm.input_size))
    with self.test_session() as sess:
      result = sess.run(output)

    self._check_output_size(btm, result)

    expected = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 2, 0, 0, 0, 0, 0, 0],
                          [3, 4, 5, 0, 0, 0, 0, 0, 0],
                          [6, 7, 8, 9, 10, 11, 0, 0, 0],
                          [12, 13, 14, 15, 16, 17, 0, 0, 0]]])
    self.assertAllEqual(result, expected)

  def test_upper(self):
    """Tests block upper-triangular matrix."""

    btm = block_matrix.BlockTriangularMatrix(
        block_shape=(2, 3), block_rows=3, upper=True)
    self.assertEqual(btm.num_blocks, 6)
    self.assertEqual(btm.block_size, 6)
    self.assertEqual(btm.input_size, 36)

    output = btm(create_input(btm.input_size))
    with self.test_session() as sess:
      result = sess.run(output)

    self._check_output_size(btm, result)

    expected = np.array([[[0, 1, 2, 3, 4, 5, 6, 7, 8],
                          [9, 10, 11, 12, 13, 14, 15, 16, 17],
                          [0, 0, 0, 18, 19, 20, 21, 22, 23],
                          [0, 0, 0, 24, 25, 26, 27, 28, 29],
                          [0, 0, 0, 0, 0, 0, 30, 31, 32],
                          [0, 0, 0, 0, 0, 0, 33, 34, 35]]])
    self.assertAllEqual(result, expected)

  def test_upper_no_diagonal(self):
    """Tests block upper-triangular matrix without diagonal."""

    btm = block_matrix.BlockTriangularMatrix(
        block_shape=(2, 3), block_rows=3, upper=True, include_diagonal=False)
    self.assertEqual(btm.num_blocks, 3)
    self.assertEqual(btm.block_size, 6)
    self.assertEqual(btm.input_size, 18)

    output = btm(create_input(btm.input_size))
    with self.test_session() as sess:
      result = sess.run(output)

    self._check_output_size(btm, result)

    expected = np.array([[[0, 0, 0, 0, 1, 2, 3, 4, 5],
                          [0, 0, 0, 6, 7, 8, 9, 10, 11],
                          [0, 0, 0, 0, 0, 0, 12, 13, 14],
                          [0, 0, 0, 0, 0, 0, 15, 16, 17],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    self.assertAllEqual(result, expected)

  def test_batch(self):
    """Tests batching."""

    btm = block_matrix.BlockTriangularMatrix(
        block_shape=(2, 2), block_rows=2, upper=False)
    output = btm(create_input(12, batch_size=2))
    with self.test_session() as sess:
      result = sess.run(output)

    self._check_output_size(btm, result, batch_size=2)

    expected = np.array([
        [[0, 1, 0, 0],
         [2, 3, 0, 0],
         [4, 5, 6, 7],
         [8, 9, 10, 11]],
        [[12, 13, 0, 0],
         [14, 15, 0, 0],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
    self.assertAllEqual(result, expected)


class BlockDiagonalMatrixTest(tf.test.TestCase):

  def test_default(self):
    """Tests BlockDiagonalMatrix."""

    bdm = block_matrix.BlockDiagonalMatrix(block_shape=(2, 3), block_rows=3)
    self.assertEqual(bdm.num_blocks, 3)
    self.assertEqual(bdm.block_size, 6)
    self.assertEqual(bdm.input_size, 18)

    output = bdm(create_input(bdm.input_size))
    with self.test_session() as sess:
      result = sess.run(output)

    expected = np.array([[[0, 1, 2, 0, 0, 0, 0, 0, 0],
                          [3, 4, 5, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 6, 7, 8, 0, 0, 0],
                          [0, 0, 0, 9, 10, 11, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 12, 13, 14],
                          [0, 0, 0, 0, 0, 0, 15, 16, 17]]])
    self.assertAllEqual(result, expected)

  def test_properties(self):
    """Tests properties of BlockDiagonalMatrix."""

    bdm = block_matrix.BlockDiagonalMatrix(block_shape=(3, 5), block_rows=7)
    self.assertEqual(bdm.num_blocks, 7)
    self.assertEqual(bdm.block_size, 15)
    self.assertEqual(bdm.input_size, 105)
    self.assertEqual(bdm.output_shape, (21, 35))
    self.assertEqual(bdm.block_shape, (3, 5))


if __name__ == "__main__":
  tf.test.main()
