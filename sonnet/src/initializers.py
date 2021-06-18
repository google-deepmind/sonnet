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
"""Initializers for Sonnet."""

import abc
import collections
from typing import Iterable, Mapping, Optional, Union
import numpy as np
from sonnet.src import types
import tensorflow as tf


class Initializer(abc.ABC):
  """Initializer base class, all initializers must implement a call method."""

  @abc.abstractmethod
  def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
    """Returns a tensor of the given ``shape`` and ``dtype``."""
    pass


class Zeros(Initializer):
  """Initializer that generates tensors initialized to 0."""

  def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
    dtype = _as_numerical_dtype(dtype)
    return tf.zeros(shape, dtype)


class Ones(Initializer):
  """Initializer that generates tensors initialized to 1."""

  def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
    dtype = _as_numerical_dtype(dtype)
    return tf.ones(shape, dtype)


class Constant(Initializer):
  """Initializer that generates tensors initialized to the given value."""

  def __init__(self, value: Union[float, int]):
    if not np.isscalar(value):
      raise TypeError("Invalid type for value: {} (expected scalar).".format(
          type(value)))
    self.value = value

  def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
    dtype = _as_numerical_dtype(dtype)
    value = tf.convert_to_tensor(self.value, dtype)
    return tf.fill(value=value, dims=shape)


class RandomUniform(Initializer):
  """Initializer that generates tensors with a uniform distribution.

  The generated values follow a uniform distribution in the range
  ``[minval, maxval)``.
  """

  def __init__(self,
               minval: types.FloatLike = 0,
               maxval: types.FloatLike = 1,
               seed: Optional[int] = None):
    """Constructs a random uniform initializer.

    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range of
        random values to generate. Defaults to ``0``.
      maxval: A python scalar or a scalar tensor. Upper bound of the range of
        random values to generate. Defaults to ``1``.
      seed: The seed used in the generation of random numbers.
    """
    self.minval = minval
    self.maxval = maxval
    self.seed = seed

  def __call__(self, shape: types.ShapeLike, dtype: tf.DType):
    dtype = _as_numerical_dtype(dtype)
    return tf.random.uniform(
        shape=shape,
        minval=self.minval,
        maxval=self.maxval,
        dtype=dtype,
        seed=self.seed)


class RandomNormal(Initializer):
  """Initializer that generates tensors with a normal distribution."""

  def __init__(self,
               mean: types.FloatLike = 0.0,
               stddev: types.FloatLike = 1.0,
               seed: Optional[int] = None):
    """Constructs a random normal initializer.

    Args:
      mean: A python scalar or a scalar tensor. Mean of the random values to
        generate.
      stddev: A python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: The seed used in the generation of random numbers.
    """
    self.mean = mean
    self.stddev = stddev
    self.seed = seed

  def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
    dtype = _as_floating_dtype(dtype)
    return tf.random.normal(
        shape=shape,
        mean=self.mean,
        stddev=self.stddev,
        dtype=dtype,
        seed=self.seed)


class TruncatedNormal(Initializer):
  """Initializer that generates a truncated normal distribution.

  These values follow a normal distribution except that values more than two
  standard deviations from the mean are discarded and re-drawn. This is the
  recommended initializer for neural network weights and filters.
  """

  def __init__(self,
               mean: types.FloatLike = 0.0,
               stddev: types.FloatLike = 1.0,
               seed: Optional[int] = None):
    """Constructs a truncated normal initializer.

    Args:
      mean: A python scalar or a scalar tensor. Mean of the random values to
        generate.
      stddev: A python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: The seed used in the generation of random numbers.
    """
    self.mean = mean
    self.stddev = stddev
    self.seed = seed

  def __call__(self, shape: types.ShapeLike, dtype: tf.DType):
    dtype = _as_floating_dtype(dtype)
    return tf.random.truncated_normal(
        shape=shape,
        mean=self.mean,
        stddev=self.stddev,
        dtype=dtype,
        seed=self.seed)


class Identity(Initializer):
  """Initializer that generates the identity matrix.

  Constructs a 2D identity matrix or batches of these.
  """

  def __init__(self, gain: float = 1.0):
    """Constructs an identity initializer.

    Args:
      gain: Multiplicative factor to apply to the identity matrix.
    """
    self.gain = gain

  def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
    dtype = _as_numerical_dtype(dtype)
    rank = shape.shape[0] if isinstance(shape, tf.Tensor) else len(shape)
    if rank < 2:
      raise ValueError("The tensor to initialize must be "
                       "at least two-dimensional")
    elif rank == 2:
      initializer = tf.eye(num_rows=shape[0], num_columns=shape[1], dtype=dtype)
    else:  # rank > 2
      initializer = tf.eye(
          num_rows=shape[-2],
          num_columns=shape[-1],
          batch_shape=shape[:-2],
          dtype=dtype)
    return self.gain * initializer


class Orthogonal(Initializer):
  """Initializer that generates an orthogonal matrix.

  NOTE: Does not support 1D tensors.

  The implementation is based on :cite:`saxe2013exact`.

  If the shape of the tensor to initialize is two-dimensional, it is initialized
  with an orthogonal matrix obtained from the QR decomposition of a matrix of
  random numbers drawn from a normal distribution.
  If the matrix has fewer rows than columns then the output will have orthogonal
  rows. Otherwise, the output will have orthogonal columns.

  If the shape of the tensor to initialize is more than two-dimensional,
  a matrix of shape ``(shape[0] * ... * shape[n - 2], shape[n - 1])``
  is initialized, where ``n`` is the length of the shape vector.
  The matrix is subsequently reshaped to give a tensor of the desired shape.
  """

  def __init__(self, gain: float = 1.0, seed: Optional[int] = None):
    """Constructs an orthogonal initializer.

    Args:
      gain: Multiplicative factor to apply to the orthogonal matrix
      seed: ``int``, the seed used in the generation of random numbers.
    """
    self.gain = gain
    self.seed = seed

  def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
    dtype = _as_floating_dtype(dtype)
    if len(shape) < 2:
      raise ValueError("The tensor to initialize must be "
                       "at least two-dimensional")
    # Flatten the input shape with the last dimension remaining
    # its original shape so it works for conv2d
    num_rows = 1
    for dim in shape[:-1]:
      num_rows *= dim
    num_cols = shape[-1]
    flat_shape = [
        tf.maximum(num_cols, num_rows),
        tf.minimum(num_cols, num_rows)
    ]

    # Generate a random matrix
    a = tf.random.normal(flat_shape, dtype=dtype, seed=self.seed)
    # Compute the qr factorization
    q, r = tf.linalg.qr(a, full_matrices=False)
    # Make Q uniform
    d = tf.linalg.tensor_diag_part(r)
    q *= tf.sign(d)
    if num_rows < num_cols:
      q = tf.linalg.matrix_transpose(q)
    return self.gain * tf.reshape(q, shape)


class VarianceScaling(Initializer):
  """Initializer capable of adapting its scale to the shape of weights tensors.

  With ``distribution="truncated_normal" or "normal"``,
  samples are drawn from a distribution with a mean of zero and a standard
  deviation (after truncation, if used) ``stddev = sqrt(scale / n)``
  where ``n`` is:

    - Number of input units in the weight tensor, if ``mode = fan_in``.
    - Number of output units, if ``mode = fan_out``.
    - Average of the numbers of input and output units, if ``mode = fan_avg``.

  Note that for transposed convolution the mode selected should be reversed. For
  number of input units use ``fan_out`` and for number of output units
  ``fan_in``.

  With ``distribution=uniform``, samples are drawn from a uniform distribution
  within ``[-limit, limit]``, with ``limit = sqrt(3 * scale / n)``.

  The variance scaling initializer can be configured to generate other standard
  initializers using the scale, mode and distribution arguments. Here are some
  example configurations:

  ==============  ==============================================================
  Name            Parameters
  ==============  ==============================================================
  glorot_uniform  scale=1.0, mode=``fan_avg``, distribution=``uniform``
  glorot_normal   scale=1.0, mode=``fan_avg``, distribution=``truncated_normal``
  lecun_uniform   scale=1.0, mode=``fan_in``,  distribution=``uniform``
  lecun_normal    scale=1.0, mode=``fan_in``,  distribution=``truncated_normal``
  he_uniform      scale=2.0, mode=``fan_in``,  distribution=``uniform``
  he_normal       scale=2.0, mode=``fan_in``,  distribution=``truncated_normal``
  ==============  ==============================================================
  """

  def __init__(self,
               scale: float = 1.0,
               mode: str = "fan_in",
               distribution: str = "truncated_normal",
               seed: Optional[int] = None):
    """Constructs a variance scaling initalizer.

    Args:
      scale: Scaling factor (positive ``float``).
      mode: One of ``fan_in``, ``fan_out``, ``fan_avg``.
      distribution: Random distribution to use. One of ``truncated_normal``,
        ``untruncated_normal`` and  ``uniform``.
      seed: ``int``, the seed used in the generation of random numbers.

    Raises:
      ValueError: In case of an invalid value for the ``scale``, ``mode`` or
        ``distribution`` arguments.
    """
    if scale <= 0.:
      raise ValueError("`scale` must be positive float.")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
      raise ValueError("Invalid `mode` argument:", mode)
    distribution = distribution.lower()
    if distribution not in {"uniform", "truncated_normal", "normal"}:
      raise ValueError("Invalid `distribution` argument:", distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed

  def __call__(self, shape: types.ShapeLike, dtype: tf.DType) -> tf.Tensor:
    dtype = _as_floating_dtype(dtype)
    scale = self.scale
    fan_in, fan_out = _compute_fans(shape)
    fan_in = tf.cast(fan_in, dtype)
    fan_out = tf.cast(fan_out, dtype)
    if self.mode == "fan_in":
      scale /= tf.maximum(1., fan_in)
    elif self.mode == "fan_out":
      scale /= tf.maximum(1., fan_out)
    else:
      scale /= tf.maximum(1., (fan_in + fan_out) / 2.)
    if self.distribution == "truncated_normal":
      # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      distribution_stddev = .87962566103423978
      stddev = tf.sqrt(scale) / distribution_stddev
      return tf.random.truncated_normal(
          shape=shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed)
    elif self.distribution == "normal":
      stddev = tf.sqrt(scale)
      return tf.random.normal(
          shape=shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed)
    else:  # self.distribution == "uniform"
      limit = tf.sqrt(3.0 * scale)
      return tf.random.uniform(
          shape=shape, minval=-limit, maxval=limit, dtype=dtype, seed=self.seed)


def check_initializers(initializers: Mapping[str, Initializer],
                       expected_keys: Iterable[str]):
  """Checks a dictionary of initializers only contains the given keys."""
  if initializers is None:
    return {}

  if not isinstance(initializers, collections.abc.Mapping):
    raise TypeError("Initializers must be a dict-like object.")

  extra_keys = frozenset(initializers) - frozenset(expected_keys)
  if extra_keys:
    raise KeyError("Invalid initializer keys {}, initializers can only "
                   "be provided for {}".format(
                       ", ".join(map(repr, extra_keys)),
                       ", ".join(map(repr, expected_keys))))

  return initializers


def _compute_fans(shape: types.ShapeLike):
  """Computes the number of input and output units for a weight shape.

  Args:
    shape: Integer shape tuple or `tf.TensorShape`.

  Returns:
    A tuple of scalars `(fan_in, fan_out)`.
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1.
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out


def _as_floating_dtype(dtype: tf.DType) -> tf.DType:
  dtype = tf.as_dtype(dtype)
  if dtype.is_floating:
    return dtype
  raise ValueError("Expected floating point type, got {}".format(dtype))


def _as_numerical_dtype(dtype: tf.DType) -> tf.DType:
  dtype = tf.as_dtype(dtype)
  if dtype.is_floating or dtype.is_integer:
    return dtype
  raise ValueError(
      "Expected integer or floating point type, got {}".format(dtype))
