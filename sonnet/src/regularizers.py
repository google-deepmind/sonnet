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
"""Regularizers for Sonnet."""

import abc
from typing import Sequence

from sonnet.src import types
import tensorflow as tf


class Regularizer(abc.ABC):
  """Base regularizer class."""

  @abc.abstractmethod
  def __call__(self, tensors: Sequence[tf.Tensor]) -> tf.Tensor:
    """Apply a regularizer.

    Args:
      tensors: A sequence of tensors to regularize.

    Returns:
      Combined regularization loss for the given tensors.
    """


class L1(Regularizer):
  """L1 regularizer.

  >>> reg = snt.regularizers.L1(0.01)
  >>> reg([tf.constant([1.0, 2.0, 3.0])])
  <tf.Tensor: ...>
  """

  def __init__(self, scale: types.FloatLike):
    """Create an L1 regularizer.

    Args:
      scale: A non-negative regularization factor.

    Raises:
      ValueError: if scale is <0.
    """
    _check_scale(scale)
    self.scale = scale

  def __repr__(self):
    # TODO(slebedev): replace with NamedTuple once we are 3.X-only.
    return "L1(scale={})".format(self.scale)

  __str__ = __repr__

  def __call__(self, tensors: Sequence[tf.Tensor]) -> tf.Tensor:
    """See base class."""
    if not tensors:
      return tf.zeros_like(self.scale)

    return self.scale * tf.add_n([tf.reduce_sum(tf.abs(t)) for t in tensors])


class L2(Regularizer):
  """L2 regularizer.

  >>> reg = snt.regularizers.L2(0.01)
  >>> reg([tf.constant([1.0, 2.0, 3.0])])
  <tf.Tensor: ...>
  """

  def __init__(self, scale: types.FloatLike):
    """Create an L2 regularizer.

    Args:
      scale: float or scalar tensor; regularization factor.

    Raises:
      ValueError: if scale is <0.
    """
    _check_scale(scale)
    self.scale = scale

  def __repr__(self):
    # TODO(slebedev): replace with NamedTuple once we are 3.X-only.
    return "L2(scale={})".format(self.scale)

  __str__ = __repr__

  def __call__(self, tensors: Sequence[tf.Tensor]) -> tf.Tensor:
    """See base class."""
    if not tensors:
      return tf.zeros_like(self.scale)

    return self.scale * tf.add_n([tf.reduce_sum(tf.square(t)) for t in tensors])


class OffDiagonalOrthogonal(Regularizer):
  """Off-diagonal orthogonal regularizer.

  The implementation is based on https://arxiv.org/abs/1809.11096.
  Given a rank N >= 2 tensor, the regularizer computes
  the sum of off-diagonal entries of (W^T W)^2 where

  * W is the input tensor reshaped to a matrix by collapsing the
    leading N - 1 axes into the first one;
  * ^2 is the element-wise square.

  NB: that is equivalent to computing the off-diagonal sum of (W^T W - I)^2,
  as off-diagonal entries of I are 0.

  For example,

      >>> t = tf.reshape(tf.range(8, dtype=tf.float32), [2, 2, 2])
      >>> reg = snt.regularizers.OffDiagonalOrthogonal(0.01)
      >>> reg([t])
      <tf.Tensor: ...>

  corresponds to copmuting

      >>> w = tf.reshape(t, [-1, 2])
      >>> w_gram_sq = tf.square(tf.matmul(tf.transpose(w), w))
      >>> 0.01 * (tf.reduce_sum(w_gram_sq) - tf.linalg.trace(w_gram_sq))
      <tf.Tensor: ...>
  """

  def __init__(self, scale: types.FloatLike):
    """Create an off-diagonal orthogonal regularizer.

    Args:
      scale: A non-negative regularization factor.

    Raises:
      ValueError: if scale is <0.
    """
    self.scale = _check_scale(scale)

  def __repr__(self):
    # TODO(slebedev): replace with NamedTuple once we are 3.X-only.
    return "Orthogonal(scale={})".format(self.scale)

  __str__ = __repr__

  def __call__(self, tensors: Sequence[tf.Tensor]) -> tf.Tensor:
    """See base class."""
    if not tensors:
      return tf.zeros_like(self.scale)

    acc = []
    for t in tensors:
      shape = t.shape.with_rank_at_least(2)
      w = tf.reshape(t, [-1, shape[-1]])
      w_gram_sq = tf.square(tf.matmul(w, w, transpose_a=True))
      # (off-diagonal sum) = (full sum) - (diagonal sum = trace).
      acc.append(tf.reduce_sum(w_gram_sq) - tf.linalg.trace(w_gram_sq))
    return self.scale * tf.add_n(acc)


def _check_scale(scale: types.FloatLike) -> types.FloatLike:
  if scale < 0:
    raise ValueError("scale must be >=0")
  return scale
