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

"""Basic Modules for TensorFlow snt.

Modules defining the simplest building blocks for Neural Networks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numbers

# Dependency imports

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from sonnet.python.modules import base
from sonnet.python.modules import util
from sonnet.python.ops import nest
import tensorflow as tf


def merge_leading_dims(tensor, n_dims=2):
  """Merge the first dimensions of a tensor.

  Args:
    tensor: Tensor to have its first dimensions merged.
    n_dims: Number of dimensions to merge.

  Returns:
    The input tensor, with its first dimensions merged.
  """
  tensor = tf.convert_to_tensor(tensor)
  tensor_shape_static = tensor.get_shape()
  tensor_shape_list = tensor_shape_static.as_list()
  if tensor_shape_static.is_fully_defined():
    new_shape = (
        [np.prod(tensor_shape_list[:n_dims])] + tensor_shape_list[n_dims:])

    return tf.reshape(tensor, new_shape)

  # Shape can't be inferred statically.
  tensor_shape = tf.shape(tensor)
  new_first_dim = tf.reduce_prod(tensor_shape[:n_dims], keep_dims=True)
  other_dims = tensor_shape[n_dims:]
  new_size = tf.concat([new_first_dim, other_dims], 0)
  result = tf.reshape(tensor, new_size)

  # We need to set the result size of this, as otherwise we won't be able to
  # pass to e.g. a Linear.
  result.set_shape([None] + tensor_shape_list[n_dims:])
  return result


def split_leading_dim(tensor, inputs, n_dims=2):
  """Split the first dimension of a tensor.

  Args:
    tensor: Tensor to have its first dimension split.
    inputs: Original reference input to look the dimensions of.
    n_dims: Number of dimensions to split.

  Returns:
    The input tensor, with its first dimension split.
  """
  input_shape_static = inputs.get_shape()
  input_shape_list = input_shape_static.as_list()
  tensor_shape_static = tensor.get_shape()
  tensor_shape_list = tensor_shape_static.as_list()
  if (input_shape_static.is_fully_defined()
      and tensor_shape_static.is_fully_defined()):
    new_shape = input_shape_list[:n_dims] + tensor_shape_list[1:]
    return tf.reshape(tensor, new_shape)

  # Shape can't be inferred statically.
  dims_after_first = tf.shape(tensor)[1:]
  split_sizes = tf.shape(inputs)[:n_dims]
  known_split_sizes = input_shape_list[:n_dims]
  known_dims_after_first = tensor_shape_list[1:]
  output_size = tf.concat([split_sizes, dims_after_first], 0)
  result = tf.reshape(tensor, output_size)
  result.set_shape(known_split_sizes + known_dims_after_first)
  return result


def create_linear_initializer(input_size):
  """Returns a default initializer for weights of a linear module."""
  stddev = 1 / math.sqrt(input_size)
  return tf.truncated_normal_initializer(stddev=stddev)


def create_bias_initializer(unused_bias_shape):
  """Returns a default initializer for the biases of a linear/AddBias module."""
  return tf.zeros_initializer()


class Linear(base.AbstractModule, base.Transposable):
  """Linear module, optionally including bias."""

  def __init__(self,
               output_size,
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               name="linear"):
    """Constructs a Linear module.

    Args:
      output_size: Output dimensionality. `output_size` can be either an integer
          or a callable. In the latter case, since the function invocation is
          deferred to graph construction time, the user must only ensure that
          output_size can be called, returning an integer, when build is called.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing initializers to initialize the
          weights (with key 'w') or biases (with key 'b'). The default
          initializer for the weights is a truncated normal initializer, which
          is commonly used when the inputs are zero centered (see
          https://arxiv.org/pdf/1502.03167v3.pdf). The default initializer for
          the bias is a zero initializer.
      partitioners: Optional dict containing partitioners to partition
          weights (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the weights
        (with key 'w') and the biases (with key 'b'). As a default, no
        regularizers are used. A regularizer should be a function that takes
        a single `Tensor` as an input and returns a scalar `Tensor` output, e.g.
        the L1 and L2 regularizers in `tf.contrib.layers`.
      name: Name of the module.

    Raises:
      KeyError: If `initializers` contains any keys other than 'w' or 'b'.
      KeyError: If `partitioners` contains any keys other than 'w' or 'b'.
      KeyError: If `regularizers` contains any keys other than 'w' or 'b'.
      TypeError: If any of the given initializers are not callable.
      TypeError: If any of the given partitioners are not callable.
      TypeError: If any of the given regularizers are not callable.
    """
    super(Linear, self).__init__(name=name)
    self._output_size = output_size
    self._use_bias = use_bias
    self._input_shape = None
    self._w = None
    self._b = None
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = util.check_initializers(
        initializers, self.possible_keys)
    self._partitioners = util.check_partitioners(
        partitioners, self.possible_keys)
    self._regularizers = util.check_regularizers(
        regularizers, self.possible_keys)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

  def _build(self, inputs):
    """Connects the Linear module into the graph, with input Tensor `inputs`.

    If this is not the first time the module has been connected to the graph,
    the Tensor provided here must have the same final dimension, in order for
    the existing variables to be the correct size for the multiplication. The
    batch size may differ for each connection.

    Args:
      inputs: A 2D Tensor of size [batch_size, input_size].

    Returns:
      A 2D Tensor of size [batch_size, output_size].

    Raises:
      base.IncompatibleShapeError: If the input is not a 2-D `Tensor` with
          the size of the second dimension specified.
      base.IncompatibleShapeError: If reconnecting an already connected module
          into the graph, and the shape of the input is not compatible with
          previous inputs.
    """
    input_shape = tuple(inputs.get_shape().as_list())

    if len(input_shape) != 2:
      raise base.IncompatibleShapeError(
          "{}: rank of shape must be 2 not: {}".format(
              self.scope_name, len(input_shape)))

    if input_shape[1] is None:
      raise base.IncompatibleShapeError(
          "{}: Input size must be specified at module build time".format(
              self.scope_name))

    if self._input_shape is not None and input_shape[1] != self._input_shape[1]:
      raise base.IncompatibleShapeError(
          "{}: Input shape must be [batch_size, {}] not: [batch_size, {}]"
          .format(self.scope_name, self._input_shape[1], input_shape[1]))

    self._input_shape = input_shape

    if "w" not in self._initializers:
      self._initializers["w"] = create_linear_initializer(self._input_shape[1])

    if "b" not in self._initializers and self._use_bias:
      self._initializers["b"] = create_bias_initializer(self._input_shape[1])

    weight_shape = (self._input_shape[1], self.output_size)
    dtype = inputs.dtype
    self._w = tf.get_variable("w",
                              shape=weight_shape,
                              dtype=dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None),
                              regularizer=self._regularizers.get("w", None))
    outputs = tf.matmul(inputs, self._w)

    if self._use_bias:
      bias_shape = (self.output_size,)
      self._b = tf.get_variable("b",
                                shape=bias_shape,
                                dtype=dtype,
                                initializer=self._initializers["b"],
                                partitioner=self._partitioners.get("b", None),
                                regularizer=self._regularizers.get("b", None))
      outputs += self._b

    return outputs

  @property
  def w(self):
    """Returns the Variable containing the weight matrix.

    Returns:
      Variable object containing the weights, from the most recent __call__.

    Raises:
      base.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._w

  @property
  def b(self):
    """Returns the Variable containing the bias.

    Returns:
      Variable object containing the bias, from the most recent __call__.

    Raises:
      base.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Linear Module when `use_bias=False`.")
    return self._b

  @property
  def output_size(self):
    """Returns the module output size."""
    if callable(self._output_size):
      self._output_size = self._output_size()
    return self._output_size

  @property
  def has_bias(self):
    """Returns `True` if bias Variable is present in the module."""
    return self._use_bias

  @property
  def initializers(self):
    """Returns the initializers dictionary."""
    return self._initializers

  @property
  def partitioners(self):
    """Returns the partitioners dictionary."""
    return self._partitioners

  @property
  def regularizers(self):
    """Returns the regularizers dictionary."""
    return self._regularizers

  def clone(self, name=None):
    """Returns a cloned `Linear` module.

    Args:
      name: Optional string assigning name of cloned module. The default name
          is constructed by appending "_clone" to `self.module_name`.

    Returns:
      Cloned `Linear` module.
    """
    if name is None:
      name = self.module_name + "_clone"
    return Linear(output_size=self.output_size,
                  use_bias=self._use_bias,
                  initializers=self._initializers,
                  partitioners=self._partitioners,
                  regularizers=self._regularizers,
                  name=name)

  # Implements Transposable interface.
  @property
  def input_shape(self):
    """Returns shape of input `Tensor` passed at last call to `build`."""
    self._ensure_is_connected()
    return self._input_shape

  # Implements Transposable interface
  def transpose(self, name=None):
    """Returns transposed `Linear` module.

    Args:
      name: Optional string assigning name of transpose module. The default name
          is constructed by appending "_transpose" to `self.module_name`.

    Returns:
      Transposed `Linear` module.
    """
    if name is None:
      name = self.module_name + "_transpose"
    return Linear(output_size=lambda: self.input_shape[1],
                  use_bias=self._use_bias,
                  initializers=self._initializers,
                  regularizers=self._regularizers,
                  name=name)


def calculate_bias_shape(input_shape, bias_dims):
  """Calculate `bias_shape` based on the `input_shape` and `bias_dims`.

  Args:
    input_shape: Shape of the input being passed into the module. The leading
        dimension is the minibatch size.
    bias_dims: The dimensions that bias should be applied over. The remaining
        dimensions will get broadcasted over.

  Returns:
    bias_shape: Tuple corresponding to the shape of bias Variable to create.

  Raises:
    ValueError: If the user attempts to add bias over the minibatch dimension,
        e.g. `bias_dims=[0]`.
  """
  input_rank = len(input_shape)
  # If None, default is to use all dimensions.
  if bias_dims is None:
    return input_shape[1:]
  # If empty list, use a scalar bias.
  elif not bias_dims:
    return ()
  # Otherwise, calculate bias_shape from bias_dims.
  else:
    bias_shape = [1] * input_rank
    # Populate bias dimensions.
    for dim in bias_dims:
      dim %= input_rank
      if dim == 0:
        raise ValueError("Cannot apply bias across the minibatch dimension.")
      bias_shape[dim] = input_shape[dim]
    # Strip leading unit dimensions.
    start = input_rank
    for dim in xrange(1, input_rank):
      if bias_shape[dim] != 1:
        start = dim
        break
    return tuple(bias_shape[start:])  # Do not apply across minibatch dimension.


class AddBias(base.AbstractModule, base.Transposable):
  """AddBias module."""

  POSSIBLE_INITIALIZER_KEYS = {"b"}

  def __init__(self,
               output_shape=None,
               bias_dims=None,
               initializers=None,
               partitioners=None,
               regularizers=None,
               name="add"):
    """Constructs an AddBias module that supports broadcasting.

    Args:
      output_shape: Output dimensionality. `output_shape` can be either `None`,
          a `tuple`, or a `callable`. In the latter case, since the function
          invocation is deferred to graph construction time, the user must only
          ensure that `output_shape` can be called, returning a tuple, when
          build is called. If `output_shape` is left as `None`, the size will be
          directly inferred by the input.
      bias_dims: List of which dimensions to retain from the input shape when
          constructing the bias. The remaining dimensions will get broadcasted
          over (given size of 1), and leading dimensions will be removed
          completely. For example, for an input of [batch_size, dim1_size,
          dim2_size, dim3_size] and `bias_dims=[1, 3]`, the resulting
          bias will have shape [dim1_size, 1, dim2_size]. The default is to
          retain all dimensions apart from the minibatch dimension. Trying to
          retain the bias shape over the minibatch dimension, e.g.
          `bias_dims=[0]`, will result in an error at build time. See the
          'Example Usage' section below for more information.
      initializers: Optional dict containing ops to initialize the biases
          (with key 'b'). The default initializer for the bias is a zero
          initializer.
      partitioners: Optional dict containing a partitioner to partition
          the bias (with key 'b'). As a default, no partitioner is used.
      regularizers: Optional dict containing regularizers of the biases
        (with key 'b'). As a default, no regularizers are used. A regularizer
        should be a function that takes a single `Tensor` as an input and
        returns a scalar `Tensor` output, e.g. the L1 and L2 regularizers in
        `tf.contrib.layers`.
      name: Name of the module.

    Example Usage:

    ```python
    # Create a 4D input Tensor.
    input = tf.random_normal(
        shape=(batch_size, dim1_size, dim2_size, dim3_size)))

    # Create a scalar bias:
    scalar_bias = snt.AddBias(bias_dims=[])
    scalar_bias_output = scalar_bias(input)
    scalar_bias.b.get_shape()  # ()

    # Create a bias over all non-minibatch dimensions:
    all_bias = snt.AddBias()  # or snt.AddBias(bias_dims=None)
    all_bias_output = all_bias(input)
    all_bias.b.get_shape()  # (dim1_size, dim2_size, dim3_size)

    # Create a bias over the last non-minibatch dimension:
    last_bias = snt.AddBias(bias_dims=[-1])
    last_bias_output = last_bias(input)
    last_bias.b.get_shape()  # (dim3_size)

    # Create a bias over the first non-minibatch dimension:
    first_bias = snt.AddBias(bias_dims=[1])
    first_bias_output = first_bias(input)
    first_bias.b.get_shape()  # (dim1_size, 1, 1)
    ```

    Raises:
      KeyError: If `initializers` contains any keys other than 'b'.
      KeyError: If `partitioners` contains any keys other than 'b'.
      KeyError: If `regularizers` contains any keys other than 'b'.
      TypeError: If any of the given initializers are not callable.
      TypeError: If any of the given partitioners are not callable.
      TypeError: If any of the given regularizers are not callable.
    """
    super(AddBias, self).__init__(name=name)
    self._output_shape = output_shape
    self._input_shape = None
    self._bias_dims = bias_dims
    self._b = None
    self._initializers = util.check_initializers(
        initializers, self.POSSIBLE_INITIALIZER_KEYS)
    self._partitioners = util.check_partitioners(
        partitioners, self.POSSIBLE_INITIALIZER_KEYS)
    self._regularizers = util.check_regularizers(
        regularizers, self.POSSIBLE_INITIALIZER_KEYS)

  def _build(self, inputs):
    """Connects the Add module into the graph, with input Tensor `inputs`.

    Args:
      inputs: A Tensor of size `[batch_size, input_size1, ...]`.

    Returns:
      A Tensor of size `[batch_size, input_size1, ...]`.

    Raises:
      base.IncompatibleShapeError: If the input is not a >= 2D `Tensor`.
      base.IncompatibleShapeError: If connecting the module into the graph
          any time after the first time, and the inferred size of the input does
          not match previous invocations.
      base.IncompatibleShapeError: If the `output_shape` has been specified
          but it does not match the input_shape`.
      base.ParentNotBuiltError: If the module is a transposed and the original
          untransposed module has not been built.
    """
    input_shape = tuple(inputs.get_shape().as_list())
    bias_shape = calculate_bias_shape(input_shape, self._bias_dims)

    # Check always contains minibatched input.
    if len(input_shape) < 2:
      raise base.IncompatibleShapeError(
          "Rank of input shape must be >=2 not: {}.".format(len(input_shape)))

    # Check previous input size is same as new input size.
    if (self._input_shape is not None and
        input_shape[1:] != self._input_shape[1:]):
      raise base.IncompatibleShapeError("Input shape has changed.")

    # If transposed, make sure that the original Module is built.
    if callable(self._output_shape):
      self._output_shape = self._output_shape()
      if self._output_shape is None:
        raise base.ParentNotBuiltError(
            "Build the original untransposed module before building this one.")

    # If output_shape specified, check that it matches input_shape.
    if (self._output_shape is not None and
        self._output_shape[1:] != input_shape[1:]):
      raise base.IncompatibleShapeError(
          "Input shape must be {} not: {}.".format(self._output_shape,
                                                   input_shape[1]))

    self._input_shape = input_shape

    if "b" not in self._initializers:
      self._initializers["b"] = create_bias_initializer(bias_shape)

    dtype = inputs.dtype
    self._b = tf.get_variable(
        "b",
        shape=bias_shape,
        dtype=dtype,
        initializer=self._initializers["b"],
        partitioner=self._partitioners.get("b", None),
        regularizer=self._regularizers.get("b", None))

    outputs = inputs + self._b
    return outputs

  @property
  def b(self):
    """Returns the Variable containing the bias.

    Returns:
      Variable object containing the bias, from the most recent __call__.

    Raises:
      base.NotConnectedError: If the module has not been connected to the
          graph yet, meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._b

  # Implements Transposable interface.
  @property
  def input_shape(self):
    """Returns shape of input `Tensor` passed at last call to `build`."""
    self._ensure_is_connected()
    return self._input_shape

  # Implements Transposable interface
  def transpose(self, name=None):
    """Returns transposed `AddBias` module.

    Args:
      name: Optional string assigning name of transpose module. The default name
          is constructed by appending "_transpose" to `self.module_name`.

    Returns:
      Transposed `AddBias` module.
    """

    if name is None:
      name = self.module_name + "_transpose"
    return AddBias(output_shape=lambda: self._input_shape,
                   bias_dims=self._bias_dims,
                   initializers=self._initializers,
                   regularizers=self._regularizers,
                   name=name)


class BatchReshape(base.AbstractModule, base.Transposable):
  """Reshapes input Tensor, preserving the batch dimension."""

  def __init__(self, shape, name="batch_reshape"):
    """Constructs a BatchReshape module.

    Args:
      shape: Shape to reshape the input Tensor to while preserving its
          batch size; `shape` can be either a tuple/list, or a callable that
          returns the actual shape. The callable does not need to be ready to
          return something meaningful at construction time, but it will be
          required to be able to do so when the module is connected to the
          graph. When the special value -1 appears in `shape` the corresponding
          size is automatically inferred. Note that -1 can only appear once in
          `shape`. To flatten all non-batch dimensions, the snt.BatchFlatten
          module can also be used.
      name: Name of the module.
    """
    super(BatchReshape, self).__init__(name=name)

    self._input_shape = None
    self._shape = shape

    if not callable(self._shape):
      self._shape = tuple(self._shape)

  def _infer_shape(self, dimensions):
    """Replaces the -1 wildcard in the output shape vector.

    This function infers the correct output shape given the input dimensions.

    Args:
      dimensions: List of input non-batch dimensions.

    Returns:
      Tuple of non-batch output dimensions.
    """
    # Size of input
    n = np.prod(dimensions)
    # Size of output where defined
    m = np.prod(abs(np.array(self._shape)))
    # Replace wildcard
    v = np.array(self._shape)
    v[v == -1] = n // m
    return tuple(v)

  def _build(self, inputs):
    """Connects the module into the graph, with input Tensor `inputs`.

    Args:
      inputs: A Tensor of shape [batch_size] + input_shape.

    Returns:
      A Tensor of shape [batch_size] + output_shape, with output_shape as
         defined in constructor.

    Raises:
      ValueError: If output shape is incompatible with input shape; or if
          shape array contains non numeric entries; or if shape array contains
          more than 1 wildcard -1.
    """
    self._input_shape = inputs.get_shape()[1:].as_list()

    if callable(self._shape):
      self._shape = tuple(self._shape())

    # Special-case 2D inputs, where no reshape is necessary. This is useful if
    # `inputs` contains empty dimensions.
    if len(self._input_shape) == 1 and len(self._shape) == 1:
      if self._shape[0] == -1 or self._shape[0] == self._input_shape[0]:
        return inputs
      else:
        raise ValueError("Output shape is incompatible with input shape")

    if not all([isinstance(x, numbers.Integral) and (x > 0 or x == -1)
                for x in self._shape]):
      raise ValueError("Input array shape can contain positive integral "
                       "numbers only, and the wildcard -1 used once")

    if self._shape.count(-1) > 1:
      raise ValueError("Wildcard -1 can appear only once in shape")

    if self._shape.count(-1) > 0:
      shape = (-1,) + self._infer_shape(self._input_shape)
    else:
      shape = (-1,) + self._shape

    if np.prod(self._input_shape) != np.prod(shape[1:]):
      raise ValueError("Output shape is incompatible with input shape")
    return tf.reshape(inputs, shape)

  @property
  def input_shape(self):
    self._ensure_is_connected()
    return self._input_shape

  # Implements Transposable interface.
  def transpose(self, name=None):
    """Returns transpose batch reshape."""
    if name is None:
      name = self.module_name + "_transpose"
    return BatchReshape(shape=lambda: self.input_shape, name=name)


class BatchFlatten(BatchReshape):
  """Flattens the input Tensor, preserving the batch dimension."""

  def __init__(self, name="batch_flatten"):
    """Constructs a BatchFlatten module.

    Args:
      name: Name of the module.
    """
    super(BatchFlatten, self).__init__(name=name, shape=(-1,))


class FlattenTrailingDimensions(BatchReshape):
  """Flattens trailing dimensions of a Tensor."""

  def __init__(self, dim_from, name="batch_dim_from"):
    """Constructs a FlattenTrailingDimensions module.

    For example, given an input Tensor with shape `[B, H, W, C, D]`, where the
    batch dimension `B` may not be statically known:

      * `dim_from=1` will return a Tensor with shape `[B, H*W*C*D]`, which
        is equivalent to `BatchFlatten`.
      * `dim_from=2` will return a Tensor with shape `[B, H, W*C*D]`.
      * `dim_from=3` will return a Tensor with shape `[B, H, W, C*D]`.
      * `dim_from=4` will return a Tensor equivalent to input.

    Args:
      dim_from: All dimensions after and including `dim_from` will
          be flattened into a single dimension.
      name: Name of the module.

    Raises:
      ValueError: If `dim_from <= 0`.
    """
    super(FlattenTrailingDimensions, self).__init__(name=name, shape=())
    if dim_from <= 0:
      raise ValueError("Argument dim_from should be >= 1.")
    self._dim_from = dim_from

  def _build(self, inputs):
    """Connects the module into the graph, with input Tensor `inputs`.

    Args:
      inputs: A Tensor of dimension at least `dim_from+1`. Only the first
        dimension may be statically unknown.

    Returns:
      A Tensor of dimension `dim_from+1`, where the size of all dimensions
          up to `dim_from` are the same as in `inputs`, and the final
          dimension has size equal to the product of the size of all dimensions
          from `dim_from`.

    Raises:
      ValueError: If `inputs` has fewer dimensions than `dim_from`.
                  If `inputs` has an statically unknown dimensions other than
                  the first.
    """

    input_shape = inputs.get_shape().as_list()
    if any([dim is None for dim in input_shape[1:]]):
      raise ValueError("Input tensor has statically unknown dimension "
                       "other than first dimension.")
    if len(input_shape) < self._dim_from + 1:
      raise ValueError("Input tensor has fewer dimensions than dim_from.")
    self._shape = tuple(input_shape[1:self._dim_from] + [-1])
    return super(FlattenTrailingDimensions, self)._build(inputs)


class TrainableVariable(base.AbstractModule):
  """Provides learnable parameter Tensor."""

  POSSIBLE_INITIALIZER_KEYS = {"w"}

  def __init__(self,
               shape,
               dtype=tf.float32,
               initializers=None,
               partitioners=None,
               regularizers=None,
               name="trainable_variable"):
    """Constructs a TrainableVariable module.

    Args:
      shape: Tensor shape.
      dtype: Tensor data type.
      initializers: Optional dictionary containing ops to initialize the weight
          Tensor, with key 'w'.
      partitioners: Optional dict containing a partitioner to partition
          the weight (with key 'w'). As a default, no partitioner is used.
      regularizers: Optional dict containing regularizers for the weights
        (with key 'w'). As a default, no regularizers are used. A regularizer
        should be a function that takes a single `Tensor` as an input and
        returns a scalar `Tensor` output, e.g. the L1 and L2 regularizers in
        `tf.contrib.layers`.
      name: Name of the module.

    Raises:
      KeyError: If `initializers` contains any keys other than 'w'.
      KeyError: If `partitioners` contains any keys other than 'w'.
      KeyError: If `regularizers` contains any keys other than 'w'.
      TypeError: If any of the given initializers are not callable.
      TypeError: If any of the given partitioners are not callable.
      TypeError: If any of the given regularizers are not callable.
    """
    super(TrainableVariable, self).__init__(name=name)

    self._shape = tuple(shape)
    self._dtype = dtype
    self._initializers = util.check_initializers(
        initializers, self.POSSIBLE_INITIALIZER_KEYS)
    self._partitioners = util.check_partitioners(
        partitioners, self.POSSIBLE_INITIALIZER_KEYS)
    self._regularizers = util.check_regularizers(
        regularizers, self.POSSIBLE_INITIALIZER_KEYS)

  def _build(self):
    """Connects the TrainableTensor module into the graph.

    Returns:
      A Tensor of shape as determined in the constructor.
    """
    if "w" not in self._initializers:
      stddev = 1 / math.sqrt(np.prod(self._shape))
      self._initializers["w"] = tf.truncated_normal_initializer(stddev=stddev)

    self._w = tf.get_variable("w",
                              shape=self._shape,
                              dtype=self._dtype,
                              initializer=self._initializers["w"],
                              partitioner=self._partitioners.get("w", None),
                              regularizer=self._regularizers.get("w", None))
    return self._w

  @property
  def w(self):
    """Returns the Variable containing the weights Tensor.

    Returns:
      Variable object containing the weights, from the most recent __call__.

    Raises:
      base.Error: If the module has not been connected to the graph yet,
          meaning the variables do not exist.
    """
    self._ensure_is_connected()
    return self._w


class BatchApply(base.AbstractModule):
  """Merges a number of leading dimensions of an input tensor to manipulate it.

  Merges a number of leading dimensions of a tensor into a single dimension,
  connects the provided module, then splits the leading dimension of the
  result to match the input.

  This is useful for applying some module to each timestep of a Time x Batch x N
  tensor. If a module is hard coded to only support 2D (Batch x N) then the
  full 3D Tensor cannot be provided. BatchApply will 'merge' the first two
  dimensions of the sequence tensor by reshaping to a (Time * Batch) x N Tensor,
  and then the internal module can be applied. The result of that operation is
  reshaped such that its first dimensions are split to match the leading
  dimensions of the input.
  """

  def __init__(self, module_or_op, n_dims=2, input_example_index=0,
               name="batch_apply"):
    """Constructor of the module.

    Args:
      module_or_op: Module or tensorflow op to apply to an input tensor.
      n_dims: Number of dimensions to merge before using module on the input
          of BatchApply.
      input_example_index: Index of input that has same shape for the first
          `n_dims` dimensions as `module_or_op` output(s). This is used for
          unflattening the output(s) if static shape inference is not possible.
      name: Name of the module.

    Raises:
      TypeError: If n_dims is not an integer.
      ValueError: If n_dims is not greater than zero.
    """
    super(BatchApply, self).__init__(name=name)
    if not isinstance(n_dims, int):
      raise TypeError("n_dims should be an integer, it is a %s instead." %
                      type(n_dims))
    if n_dims <= 0:
      raise ValueError("n_dims should be greater than zero.")
    self._module = module_or_op
    self._n_dims = n_dims
    self._input_example_index = input_example_index

  def _build(self, *args):
    """Connects the BatchApply module into the graph.

    Args:
      *args: a Tensor or a nested list of Tensors. The input tensors will
          have their first dimensions merged, then an op or a module will be
          called on the input. The first dimension of the output will be
          split again based on the leading dimensions of the first input
          tensor.

    Returns:
      A Tensor resulting of applying the process above.
    """
    # Merge leading dimensions for each input Tensor, then apply inner module.
    merged = nest.map(lambda inp: merge_leading_dims(inp, self._n_dims),
                      args)
    results = self._module(*merged)

    # Unmerging takes the sizes of the leading dimensions from an input example
    # with equal shape for the leading `n_dims` dimensions. Typically this is
    # the first input.
    example_input = tf.convert_to_tensor(
        nest.flatten(args)[self._input_example_index])
    def _split_to_original_leading_dims(result):
      return split_leading_dim(result, example_input, self._n_dims)
    return nest.map(_split_to_original_leading_dims, results)


class SliceByDim(base.AbstractModule):
  """Slices a tensor along specific dimensions.

  The user can slice a tensor by specifying only the list of dimensions that
  they want to slice, together with the lists of integers containing the
  beginning indices of the slicing, and the size of the slices. Hence, with
  `SliceByDim` slicing can be performed without knowing in advance the rank of
  the input tensor.

  Tensorflow also offers a built-in op performing slicing, `tf.slice`. However,
  `tf.slice` requires all the slicing dimensions to be specified, using
  wildcards when no slicing is required. For example, with `tf.slice`, slicing
  half a 5D tensor along dimension `1` would be:

  ```python
  output = tf.slice(inputs,
                    begin=[0, 0, 0, 0, 0],
                    size=[-1, inputs.get_shape()[1].value//2, -1, -1, -1])
  ```

  The same operation using `SliceByDim` would be:

  ```python
  output = SliceByDim(dims=[1], begin=[0], size=[x.get_shape()[1].value//2])(x)
  ```

  `SliceByDim` can be used to specify multiple slicing dimensions, for example:

  ```python
  output = SliceByDim(dims=[1, 3], begin=[0, 0], size=[12, 24])(x)
  ```
  """

  def __init__(self, dims, begin, size, name="slice_by_dim"):
    """Constructs the `SliceByDim` module.

    Args:
      dims: The dimensions to slice along, as a list of unique integers.
          Negative integers index from the final dimension backwards, as in
          python arrays.
      begin: The beginning indices of the slicing, as a list of integers. Must
          be the same length as the `dims` list.
      size: The size of the slices, as a list of integers. Must be the same
          length as the `dims` list.
      name: The name of the module.

    Raises:
      ValueError: If `dims` has non-unique integers, or if the size of `begin`
          is different from the size of `dims`, or if the size of `size` is
          different from the size of `dims`.
    """
    super(SliceByDim, self).__init__(name=name)
    self._dims = dims
    self._begin = begin
    self._size = size
    if np.unique(dims).size != len(dims):
      raise ValueError("dims must not have any repeated integers.")
    if len(begin) != len(dims):
      raise ValueError(
          "begin must have the same length as dims: {}.".format(len(dims)))
    if len(size) != len(dims):
      raise ValueError(
          "size must have the same length as dims: {}.".format(len(dims)))

  def _build(self, inputs):
    """Connects the SliceByDim module into the graph.

    Args:
      inputs: `Tensor` to slice. Its rank must be greater than the maximum
          dimension specified in `dims` (plus one as python is 0 indexed).

    Returns:
      The sliced tensor.

    Raises:
      ValueError: If `inputs` tensor has insufficient rank.
    """
    shape_inputs = inputs.get_shape().as_list()
    rank = len(shape_inputs)

    # Checks that the rank of the tensor.
    max_dim = np.max(self._dims) + 1
    if rank < max_dim:
      raise ValueError("Rank of inputs must be at least {}.".format(max_dim))

    # Builds default lists for begin and size to pass to `tf.slice`.
    full_begin = [0] * rank
    full_size = [-1] * rank

    # Updates lists with what the user provided.
    for dim, begin, size in zip(self._dims, self._begin, self._size):
      full_begin[dim] = begin
      full_size[dim] = size

    return tf.slice(inputs, begin=full_begin, size=full_size)


class TileByDim(base.AbstractModule):
  """Tile a tensor along specific dimensions.

  The user can tile a tensor by specifying only the list of dimensions that
  they want to tile, together with the lists of integers containing the
  multiples of the tiling. Hence, with `TileByDim` tiling can be performed
  without knowing in advance the rank of the input tensor.

  Tensorflow also offers a built-in op performing tiling, `tf.tile`. However,
  `tf.tile` requires all the tiling dimensions to be specified, using `1`
  when no tiling is required. For example, with tf.tiling, tiling a 5D
  tensor along dimension `1`, by `2` would be:

  ```python
  output = tf.tile(inputs, multiples=[1, 2, 1, 1, 1])
  ```

  The same operation using `TileByDim` would be:

  ```python
  output = TileByDim(dims=[1], multiples=[2])(x)
  ```

  `TileByDim` can be used to specify multiple tiling dimensions, for example:

  ```python
  output = TileByDim(dims=[1, 3], multiples=[2, 4])(x)
  ```
  """

  def __init__(self, dims, multiples, name="tile_by_dim"):
    """Constructs the `TileByDim` module.

    Args:
      dims: The dimensions to tile along, as a list of unique integers.
      multiples: The multiple of the tiling, as a list of integers. Must
          be the same length as the `dims` list.
      name: The name of the module.

    Raises:
      ValueError: If `dims` has non-unique integers, or if the size of
          `multiples` is different from the size of `dims`.
    """
    super(TileByDim, self).__init__(name=name)
    self._dims = dims
    self._multiples = multiples
    if np.unique(dims).size != len(dims):
      raise ValueError("dims must not have any repeated integers.")
    if len(multiples) != len(dims):
      raise ValueError(
          "multiples must have the same length as dims: {}.".format(len(dims)))

  def _build(self, inputs):
    """Connects the `TileByDim` module into the graph.

    Args:
      inputs: `Tensor` to tile.

    Returns:
      The tiled tensor.
    """
    shape_inputs = inputs.get_shape().as_list()
    rank = len(shape_inputs)

    # Builds default lists for multiples to pass to `tf.tile`.
    full_multiples = [1] * rank

    # Updates lists with what the user provided.
    for dim, multiple in zip(self._dims, self._multiples):
      full_multiples[dim] = multiple

    return tf.tile(inputs, multiples=full_multiples)


class MergeDims(base.AbstractModule):
  """Merges a tensor or nested list of tensors along a range of dimensions.

  Tensors are reshaped by specifying the range of dimensions to merge.
  Hence, the reshape can be performed without knowing in advance the rank of
  the input tensor.

  For example, merging dimensions 1, 2 and 3 together can be performed by
  calling:

  output = MergeDims(start=1, size=3)(x)

  A nested list of tensors can be merged:

  x = [tf.random_uniform(shape=[5, 5]), [tf.random_uniform(shape=[3, 3, 3])]]
  output = MergeDims(start=0, size=2)(x)
  """

  def __init__(self, start, size, name="merge_dims"):
    """Constructs the MergeDims module.

    Args:
      start: Start of the range of dimensions to merge.
      size: Size the range of dimensions to merge.
      name: The name of the module.

    Raises:
      ValueError: If `size` is not strictly greater than 1.
    """
    super(MergeDims, self).__init__(name=name)
    self._start = start
    self._size = size

    # Checks for non consecutive integers.
    if size <= 1:
      raise ValueError("`size` should be strictly greater than 1.")

  def _merge(self, tensor):
    output_shape = tensor.get_shape().as_list()
    rank = len(output_shape)
    if rank < self._start + self._size:
      raise ValueError("Rank of inputs must be at least {}."
                       .format(self._start + self._size))

    # Update the shape of the merged dimensions.
    output_shape[self._start:self._start + self._size] = [-1]

    return tf.reshape(tensor, shape=output_shape)

  def _build(self, inputs):
    """Connects the MergeDims module into the graph.

    Args:
      inputs: Tensor or a nested list of Tensors to merge. Its rank must be
          greater than or equal to `start` + `size`.

    Returns:
      The merged Tensor or a nested list of merged Tensors.

    Raises:
      ValueError: If any of the `inputs` tensors has insufficient rank.
    """
    if nest.is_sequence(inputs):
      merged_tensors = [self._merge(tensor) for tensor in nest.flatten(inputs)]
      return nest.pack_sequence_as(inputs, merged_tensors)

    # inputs is a single tf.Tensor
    return self._merge(inputs)


class SelectInput(base.AbstractModule):
  """Returns a subset of its inputs in an arbitrarily nested configuration.

  This module can be used for multiple purposes.

  The basic usage is to select a tensor or a subset of tensors:

  ```
  output = snt.SelectInput(idx=0, name='select')(input0, input1)
  ==> input0

  output = snt.SelectInput(idx=[0, 2], name='select')(input0, input1, input2)
  ==> (input0, input2)
  ```

  Another usage is to change the orders of the input tensors:

  ```
  output = snt.SelectInput(idx=[1, 0], name='select')(input0, input1)
  ==> (input1, input0)
  ```

  Another usage is to duplicate an input:

  ```
  output = snt.SelectInput(idx=[0, 0], name='select')(input0)
  ==> (input0, input0)
  ```

  Another usage is to add arbitrary nesting:

  ```
  output = snt.SelectInput(
      idx=[0, [1, [2]]], name='select')(input0, input1, input2)
  ==> (input0, (input1, (input2,)))
  ```
  """

  def __init__(self, idx, name="select_input"):
    """Module constructor.

    Args:
      idx: Indexes of the tensors to select. If `idx` is an integer, then
          a `Tensor` is returned. If `idx` is a (nested) list/tuple, then a
          (nested) tuple of `Tensor` is returned.
      name: Name of the module.

    Raises:
      TypeError: If `idx` is not an list, tuple or integer.
    """
    super(SelectInput, self).__init__(name=name)
    self._check_type(idx)
    self._idx = idx

  def _check_type(self, idx):
    if isinstance(idx, (list, tuple)):
      for value in idx:
        self._check_type(value)
    elif not isinstance(idx, int):
      raise TypeError("`idx` should be a (nested) array/tuple, or an integer.")

  def _select(self, inputs, idx):
    if isinstance(idx, (list, tuple)):
      return tuple(self._select(inputs, i) for i in idx)
    else:
      if idx < 0 or idx >= len(inputs):
        raise ValueError("`idx` contains out of bound entries (they should be "
                         "in the range [0, {}))".format(len(inputs)))
      # Identity is called otherwise we might get 'placeholder is both fed and
      # fetched' errors in some cases when using a feed_dict.
      return tf.identity(inputs[idx])

  def _build(self, *inputs):
    """Connects the module into the graph.

    Args:
      *inputs: `Tensor` variables to select.

    Returns:
      Subset of `inputs` in an arbitrarily nested configuration.

    Raises:
      ValueError: If any entry of `idx` is out of bounds with respect to the
          size of `inputs`.
    """
    return self._select(inputs, self._idx)
