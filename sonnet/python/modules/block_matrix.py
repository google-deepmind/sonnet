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

"""Modules for dealing with block matrices."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from six.moves import xrange  # pylint: disable=redefined-builtin
from sonnet.python.modules import base
import tensorflow as tf


class BlockTriangularMatrix(base.AbstractModule):
  """Module for constructing a block triangular matrix from a vector.

  This module takes a vector and builds a block (upper or lower) triangular
  matrix from it. The blocks have equal shape, `block_shape`, and the number of
  rows (and, hence, the number of columns) needs to be specified in advance.
  The diagonal may be excluded by setting the argument `include_diagonal`
  to False.

  Example: suppose that we choose `block_shape = (2, 2)` and
  `block_rows = 3`. Then, the input vector `[1 2 3 ... 24]` is mapped to
  the matrix:

  ```
  M = [ 1  2  0  0  0  0
        3  4  0  0  0  0
        5  6  7  8  0  0
        9 10 11 12  0  0
       13 14 15 16 17 18
       19 20 21 22 23 24].
  ```
  """

  def __init__(self,
               block_shape,
               block_rows,
               include_diagonal=True,
               include_off_diagonal=True,
               upper=False,
               name='block_triangular_matrix'):
    """Constructs a new `BlockTriangularMatrix` module.

    Args:
      block_shape: tuple, 2-dimensional tuple indicating the shape of each
        individual block.
      block_rows: int, the number of blocks in each row (and column) of the
        output matrix.
      include_diagonal: boolean, indicates whether or not blocks on the diagonal
        entries should be included.
      include_off_diagonal: boolean, indicates whether or not only the
        off-diagonal entries should be included. If set to False, the value of
        `upper` is ignored.
      upper: boolean, if True then the output matrix is block upper triangular;
        if False, it is block lower triangular.
      name: string, name of the module.

    Raises:
      ValueError: if `include_diagonal` and `include_off_diagonal` are both
        False.
    """
    super(BlockTriangularMatrix, self).__init__(name=name)
    if not include_diagonal and not include_off_diagonal:
      raise ValueError('Arguments include_diagonal and include_off_diagonal '
                       'cannot both be False.')

    self._block_shape = tuple(block_shape)
    self._block_rows = block_rows
    self._include_diagonal = include_diagonal
    self._include_off_diagonal = include_off_diagonal
    self._upper = upper
    self._num_blocks = sum(
        self._content_blocks(r) for r in xrange(self._block_rows))

  @property
  def num_blocks(self):
    """The total number of blocks in the output matrix."""
    return self._num_blocks

  @property
  def block_size(self):
    """The number of entries of each block."""
    return self._block_shape[0] * self._block_shape[1]

  @property
  def block_shape(self):
    """The shape of each block."""
    return self._block_shape

  @property
  def output_shape(self):
    """The shape of the output matrix."""
    return (self._block_shape[0] * self._block_rows,
            self._block_shape[1] * self._block_rows)

  @property
  def input_size(self):
    """The expected length of the input vector."""
    return self.block_size * self.num_blocks

  def _build(self, vector):
    vector.get_shape().assert_is_compatible_with((None, self.input_size))
    n = tf.shape(vector)[0]  # Get batch size.

    rows = []
    start_index = 0
    block_height, block_width = self._block_shape

    # Construct the individual block rows.
    for r in xrange(self._block_rows):
      # Construct an individual block row as a concatenation of a block of
      # zeros (left zeros), the actual content (coming from the input), and
      # another block of zeros (right zeros). Each of these blocks can be empty.
      left_zero_blocks = self._left_zero_blocks(r)
      right_zero_blocks = self._right_zero_blocks(r)
      content_blocks = self._content_blocks(r)

      assert (left_zero_blocks + content_blocks + right_zero_blocks
              == self._block_rows)

      assert left_zero_blocks >= 0
      assert right_zero_blocks >= 0
      assert content_blocks >= 0

      # Take the next chunk of entries from the input vector
      # and increase the starting index into the input vector.
      end_index = start_index + content_blocks * self.block_size
      input_chunk = vector[:, start_index:end_index]
      start_index = end_index

      # Reshape the entries from the input vector.
      content = tf.reshape(
          input_chunk,
          shape=(n, block_height, content_blocks * block_width),
          name='content' + str(r))
      paddings = [[0, 0], [0, 0],
                  [left_zero_blocks * block_width,
                   right_zero_blocks * block_width]]
      # Concatenate content and zeros to form the next block row.
      rows.append(tf.pad(content, paddings, name='block_row' + str(r)))

    # Concatenate all rows together to get the final block matrix.
    return tf.concat(rows, 1)

  def _left_zero_blocks(self, r):
    """Number of blocks with zeros from the left in block row `r`."""
    if not self._include_off_diagonal:
      return r
    elif not self._upper:
      return 0
    elif self._include_diagonal:
      return r
    else:
      return r + 1

  def _right_zero_blocks(self, r):
    """Number of blocks with zeros from the right in block row `r`."""
    if not self._include_off_diagonal:
      return self._block_rows - r - 1
    elif self._upper:
      return 0
    elif self._include_diagonal:
      return self._block_rows - r - 1
    else:
      return self._block_rows - r

  def _content_blocks(self, r):
    """Number of content blocks in block row `r`."""
    return (self._block_rows - self._left_zero_blocks(r)
            - self._right_zero_blocks(r))


class BlockDiagonalMatrix(BlockTriangularMatrix):
  """Module for constructing a block diagonal matrix from a vector.

  This module takes a vector and builds a block diagonal matrix from
  it. The blocks have equal shape, `block_shape`, and the number of rows
  (and, hence, the number of columns) needs to be specified in advance.

  Example: suppose that we choose `block_shape = (2, 2)` and
  `block_rows = 3`. Then, the input vector `[1 2 3 ... 12]` is mapped to
  the matrix:

  ```
  M = [ 1  2  0  0  0  0
        3  4  0  0  0  0
        0  0  5  6  0  0
        0  0  7  8  0  0
        0  0  0  0  9 10
        0  0  0  0 11 12].
  ```
  """

  def __init__(self,
               block_shape,
               block_rows,
               name='block_diagonal_matrix'):
    """Constructs a new `BlockDiagonalMatrix` module.

    Args:
      block_shape: tuple, 2-dimensional tuple indicating the shape of each
        individual block.
      block_rows: int, the number of blocks in each row (and column) of the
        output matrix.
      name: string, name of the module.
    """
    super(BlockDiagonalMatrix, self).__init__(
        block_shape=block_shape,
        block_rows=block_rows,
        include_diagonal=True,
        include_off_diagonal=False,
        name=name)
