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

"""Implementation of convolutional Sonnet modules.

Classes defining convolutional operations, inheriting from `snt.Module`, with
easy weight sharing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numbers

# Dependency imports

import numpy as np
import six
from sonnet.python.modules import base
from sonnet.python.modules import util
import tensorflow as tf


# Strings for TensorFlow convolution padding modes. See the following
# documentation for an explanation of VALID versus SAME:
# https://www.tensorflow.org/api_guides/python/nn#Convolution
SAME = "SAME"
VALID = "VALID"
FULL = "FULL"
CAUSAL = "CAUSAL"
REVERSE_CAUSAL = "REVERSE_CAUSAL"
CONV_OP_ALLOWED_PADDINGS = {SAME, VALID}
ALLOWED_PADDINGS = {SAME, VALID, FULL, CAUSAL, REVERSE_CAUSAL}

DATA_FORMAT_NCW = "NCW"
DATA_FORMAT_NWC = "NWC"
SUPPORTED_1D_DATA_FORMATS = {DATA_FORMAT_NCW, DATA_FORMAT_NWC}

DATA_FORMAT_NCHW = "NCHW"
DATA_FORMAT_NHWC = "NHWC"
SUPPORTED_2D_DATA_FORMATS = {DATA_FORMAT_NCHW, DATA_FORMAT_NHWC}

DATA_FORMAT_NDHWC = "NDHWC"
DATA_FORMAT_NCDHW = "NCDHW"
SUPPORTED_3D_DATA_FORMATS = {DATA_FORMAT_NDHWC, DATA_FORMAT_NCDHW}


def _default_transpose_size(input_shape, stride, kernel_shape=None,
                            padding=SAME):
  """Returns default (maximal) output shape for a transpose convolution.

  In general, there are multiple possible output shapes that a transpose
  convolution with a given `input_shape` can map to. This function returns the
  output shape which evenly divides the stride to produce the input shape in
  a forward convolution, i.e. the maximal valid output shape with the given
  configuration:

  if the padding type is SAME then:  output_shape = input_shape * stride
  if the padding type is VALID then: output_shape = input_shape * stride +
                                                    kernel_shape - 1

  See the following documentation for an explanation of VALID versus SAME
  padding modes:
  https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#convolution

  Args:
    input_shape: Sequence of sizes of each dimension of the input, excluding
      batch and channel dimensions.
    stride: Sequence or integer of kernel strides, excluding batch and channel
      dimension strides.
    kernel_shape: Sequence or integer of kernel sizes.
    padding: Padding algorithm, either `snt.SAME` or `snt.VALID`.

  Returns:
    output_shape: A tuple of sizes for a transposed convolution that divide
      evenly with the given strides, kernel shapes, and padding algorithm.
  """
  if not input_shape:
    raise TypeError("input_shape is None; if using Sonnet, are you sure you "
                    "have connected the module to inputs?")
  input_length = len(input_shape)
  stride = _fill_and_verify_parameter_shape(stride, input_length, "stride")
  padding = _verify_conv_op_supported_padding(padding)

  output_shape = tuple(x * y for x, y in zip(input_shape, stride))

  if padding == VALID:
    kernel_shape = _fill_and_verify_parameter_shape(kernel_shape, input_length,
                                                    "kernel")
    output_shape = tuple(x + y - 1 for x, y in zip(output_shape, kernel_shape))

  return output_shape


def _fill_shape(x, n):
  """Converts a dimension to a tuple of dimensions of a given size.

  This is used to allow shorthand notation for various configuration parameters.
  A user can provide either, for example, `2` or `[2, 2]` as a kernel shape, and
  this function returns `(2, 2)` in both cases. Passing `[1, 2]` will return
  `(1, 2)`.

  Args:
    x: An integer, tf.Dimension, or an iterable of them.
    n: An integer, the size of the desired output list

  Returns:
    If `x` is an integer, a tuple of size `n` containing `n` copies of `x`.
    If `x` is an iterable of integers or tf.Dimension of size `n`, it returns
      `tuple(x)`.

  Raises:
    TypeError: If n is not a positive integer;
      or if x is neither integer nor an iterable of size n.
  """
  if not isinstance(n, numbers.Integral) or n < 1:
    raise TypeError("n must be a positive integer")

  if (isinstance(x, numbers.Integral) or isinstance(x, tf.Dimension)) and x > 0:
    return (x,) * n

  try:
    if len(x) == n and all(v > 0 for v in x):
      return tuple(x)
  except TypeError:
    pass

  raise TypeError("x is {}, must be either a positive integer "
                  "or an iterable of positive integers of size {}"
                  .format(x, n))


def _fill_and_verify_parameter_shape(x, n, parameter_label):
  """Expands x if necessary into a `n`-D kernel shape and reports errors."""
  try:
    return _fill_shape(x, n)
  except TypeError as e:
    raise base.IncompatibleShapeError("Invalid " + parameter_label + " shape: "
                                      "{}".format(e))


def _verify_conv_op_supported_padding(padding):
  """Verifies that the given padding type is supported for conv ops.

  Args:
    padding: One of CONV_OP_ALLOWED_PADDINGS.

  Returns:
    padding.

  Raises:
    ValueError: If padding is not one of CONV_OP_ALLOWED_PADDINGS.
  """
  if padding not in CONV_OP_ALLOWED_PADDINGS:
    raise ValueError(
        "Padding must be member of '{}', not {}".format(
            CONV_OP_ALLOWED_PADDINGS, padding))
  return padding


def _fill_and_verify_padding(padding, n):
  """Verifies that the provided padding is supported and expands to size n.

  Args:
    padding: One of ALLOWED_PADDINGS, or an iterable of them.
    n: An integer, the size of the desired output list.

  Returns:
    If `padding` is one of ALLOWED_PADDINGS, a tuple of size `n` containing `n`
    copies of `padding`.
    If `padding` is an iterable of ALLOWED_PADDINGS of size `n`, it returns
    `padding(x)`.

  Raises:
    TypeError: If n is not a positive integer; if padding is neither one of
      ALLOWED_PADDINGS nor an iterable of ALLOWED_PADDINGS of size n.
  """
  if not isinstance(n, numbers.Integral) or n < 1:
    raise TypeError("n must be a positive integer")

  if isinstance(padding, six.string_types) and padding in ALLOWED_PADDINGS:
    return (padding,) * n

  try:
    if len(padding) == n and all(p in ALLOWED_PADDINGS for p in padding):
      return tuple(padding)
  except TypeError:
    pass

  raise TypeError("padding is {}, must be member of '{}' or an iterable of "
                  "these of size {}".format(padding, ALLOWED_PADDINGS, n))


def _padding_to_conv_op_padding(padding):
  """Whether to use SAME or VALID for the underlying convolution op.

  Args:
    padding: A tuple of members of ALLOWED_PADDINGS, e.g. as returned from
      `_fill_and_verify_padding`.

  Returns:
    One of CONV_OP_ALLOWED_PADDINGS, the padding method to use for the
    underlying convolution op.

  Raises:
    ValueError: If padding is not a tuple.
  """
  if not isinstance(padding, tuple):
    raise ValueError("padding should be a tuple.")
  if all(p == SAME for p in padding):
    # If we want SAME padding for all dimensions then we can use SAME for the
    # conv and avoid doing any extra padding.
    return SAME
  else:
    # Otherwise we prefer to use VALID, since we can implement all the other
    # padding types just by adding some extra padding before doing a VALID conv.
    # (We could use SAME but then we'd also have to crop outputs in some cases).
    return VALID


def _fill_and_one_pad_stride(stride, n, data_format=DATA_FORMAT_NHWC):
  """Expands the provided stride to size n and pads it with 1s."""
  if isinstance(stride, numbers.Integral) or (
      isinstance(stride, collections.Iterable) and len(stride) <= n):
    if data_format.startswith("NC"):
      return (1, 1,) + _fill_shape(stride, n)
    elif data_format.startswith("N") and data_format.endswith("C"):
      return (1,) + _fill_shape(stride, n) + (1,)
    else:
      raise ValueError(
          "Invalid data_format {:s}. Must start with N and have a channel dim "
          "either follow the N dim or come at the end".format(data_format))
  elif isinstance(stride, collections.Iterable) and len(stride) == n + 2:
    return stride
  else:
    raise base.IncompatibleShapeError(
        "stride is {} ({}), must be either a positive integer or an iterable of"
        " positive integers of size {}".format(stride, type(stride), n))


def _verify_inputs(inputs, channel_index, data_format):
  """Verifies `inputs` is semantically correct.

  Args:
    inputs: An input tensor provided by the user.
    channel_index: The index of the channel dimension.
    data_format: The format of the data in `inputs`.

  Raises:
    base.IncompatibleShapeError: If the shape of `inputs` doesn't match
      `data_format`.
    base.UnderspecifiedError: If the channel dimension of `inputs` isn't
      defined.
    TypeError: If input Tensor dtype is not compatible with either
      `tf.float16`, `tf.bfloat16`, `tf.float32` or `tf.float64`.
  """
  # Check shape.
  input_shape = tuple(inputs.get_shape().as_list())
  if len(input_shape) != len(data_format):
    raise base.IncompatibleShapeError((
        "Input Tensor must have rank {} corresponding to "
        "data_format {}, but instead was {} of rank {}.").format(
            len(data_format), data_format, input_shape, len(input_shape)))

  # Check type.
  if not (tf.float16.is_compatible_with(inputs.dtype) or
          tf.bfloat16.is_compatible_with(inputs.dtype) or
          tf.float32.is_compatible_with(inputs.dtype) or
          tf.float64.is_compatible_with(inputs.dtype)):
    raise TypeError(
        "Input must have dtype tf.float16, tf.bfloat16, tf.float32 or "
        "tf.float64, but dtype was {}".format(inputs.dtype))

  # Check channel dim.
  input_channels = input_shape[channel_index]
  if input_channels is None:
    raise base.UnderspecifiedError(
        "Number of input channels must be known at module build time")


def create_weight_initializer(fan_in_shape, dtype=tf.float32):
  """Returns a default initializer for the weights of a convolutional module."""
  stddev = 1 / math.sqrt(np.prod(fan_in_shape))
  return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)


def create_bias_initializer(unused_bias_shape, dtype=tf.float32):
  """Returns a default initializer for the biases of a convolutional module."""
  return tf.zeros_initializer(dtype=dtype)


def _find_channel_index(data_format):
  """Returns the index of the channel dimension.

  Args:
    data_format: A string of characters corresponding to Tensor dimensionality.

  Returns:
    channel_index: An integer indicating the channel dimension.

  Raises:
    ValueError: If no channel dimension was found.
  """
  for i, c in enumerate(data_format):
    if c == "C":
      return i
  raise ValueError("data_format requires a channel dimension. Got: {}"
                   .format(data_format))


def _apply_bias(inputs, outputs, channel_index, data_format, output_channels,
                initializers, partitioners, regularizers):
  """Initialize and apply a bias to the outputs.

  Figures out the shape of the bias vector, initialize it, and applies it.

  Args:
    inputs: A Tensor of shape `data_format`.
    outputs: A Tensor of shape `data_format`.
    channel_index: The index of the channel dimension in `inputs`.
    data_format: Format of `inputs`.
    output_channels: Channel dimensionality for `outputs`.
    initializers: Optional dict containing ops to initialize the biases
      (with key 'b').
    partitioners: Optional dict containing partitioners to partition the
      biases (with key 'b').
    regularizers: Optional dict containing regularizers for the biases
      (with key 'b').

  Returns:
    b: The constructed bias variable.
    outputs: The `outputs` argument that has had a bias applied.
  """
  bias_shape = (output_channels,)
  if "b" not in initializers:
    initializers["b"] = create_bias_initializer(bias_shape,
                                                dtype=inputs.dtype)
  b = tf.get_variable("b",
                      shape=bias_shape,
                      dtype=inputs.dtype,
                      initializer=initializers["b"],
                      partitioner=partitioners.get("b", None),
                      regularizer=regularizers.get("b", None))

  # tf.nn.bias_add only supports 2 data formats.
  if data_format in (DATA_FORMAT_NHWC, DATA_FORMAT_NCHW):
    # Supported as-is.
    outputs = tf.nn.bias_add(outputs, b, data_format=data_format)
  else:
    # Create our own bias vector.
    bias_correct_dim = [1] * len(data_format)
    bias_correct_dim[channel_index] = output_channels
    outputs += tf.reshape(b, bias_correct_dim)

  return b, outputs


class _ConvND(base.AbstractModule):
  """N-dimensional convolution and dilated convolution module, including bias.

  This acts as a light wrapper around the TensorFlow ops `tf.nn.convolution`
  abstracting away variable creation and sharing.
  """

  def __init__(self, output_channels, kernel_shape, stride=1, rate=1,
               padding=SAME, use_bias=True, initializers=None,
               partitioners=None, regularizers=None, mask=None,
               data_format=DATA_FORMAT_NHWC,
               custom_getter=None, name="conv_nd"):
    """Constructs a _ConvND module.

    Args:
      output_channels: Number of output channels. `output_channels` can be
          either a number or a callable. In the latter case, since the function
          invocation is deferred to graph construction time, the user must only
          ensure that output_channels can be called, returning an integer,
          when `build` is called.
      kernel_shape: Sequence of kernel sizes (up to size N), or an integer.
          `kernel_shape` will be expanded to define a kernel size in all
          dimensions.
      stride: Sequence of strides (up to size N), or an integer.
          `stride` will be expanded to define stride in all dimensions.
      rate: Sequence of dilation rates (of size N), or integer that is used to
          define dilation rate in all dimensions. 1 corresponds to standard ND
          convolution, `rate > 1` corresponds to dilated convolution. Cannot be
          > 1 if any of `stride` is also > 1.
      padding: Padding algorithm. Either `snt.SAME`, `snt.VALID`, `snt.FULL`,
          `snt.CAUSAL`, `snt.REVERSE_CAUSAL`, or a sequence of these paddings
          (up to size N).
          * snt.SAME and snt.VALID are explained in the Tensorflow docs at
            https://www.tensorflow.org/api_guides/python/nn#Convolution.
          * snt.FULL pre- and post-pads with the maximum padding which does not
            result in a convolution over just padded elements.
          * snt.CAUSAL pre-pads to ensure that each output value only depends on
            input values at the same or preceding indices ("no dependence on the
            future").
          * snt.REVERSE_CAUSAL post-pads to ensure that each output value only
            depends on input values at the same or *greater* indices ("no
            dependence on the past").
          If you use the same padding for all dimensions, and it is one of SAME
          or VALID, then this is supported directly by the underlying
          convolution op. In all other cases, the input data will be padded
          using tf.pad before calling the convolution op.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializer for the
          weights is a truncated normal initializer, which is commonly used
          when the inputs are zero centered (see
          https://arxiv.org/pdf/1502.03167v3.pdf). The default initializer for
          the bias is a zero initializer.
      partitioners: Optional dict containing partitioners to partition
          weights (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes
          a single `Tensor` as an input and returns a scalar `Tensor` output,
          e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      mask: A convertible to a ND tensor which is multiplied
          component-wise with the weights (Optional).
      data_format: The data format of the input.
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      base.IncompatibleShapeError: If the given kernel shape is not an integer;
          or if the given kernel shape is not a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is not an integer; or if
          the given stride is not a sequence of two integers.
      base.IncompatibleShapeError: If the given rate is not an integer; or if
          the given rate is not a sequence of two integers.
      base.IncompatibleShapeError: If a mask is a TensorFlow Tensor with
          a not fully defined shape.
      base.NotSupportedError: If rate in any dimension and the stride in any
          dimension are simultaneously > 1.
      ValueError: If the given padding is not `snt.VALID`, `snt.SAME`,
          `snt.FULL`, `snt.CAUSAL`, `snt.REVERSE_CAUSAL` or a sequence of these.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      TypeError: If mask is given and it is not convertible to a Tensor.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
    """
    super(_ConvND, self).__init__(custom_getter=custom_getter, name=name)

    self._n = len(data_format) - 2
    self._input_channels = None
    self._output_channels = output_channels
    self._kernel_shape = _fill_and_verify_parameter_shape(kernel_shape, self._n,
                                                          "kernel")
    self._data_format = data_format

    # The following is for backwards-compatibility from when we used to accept
    # N-strides of the form [1, ..., 1].
    if (isinstance(stride, collections.Iterable) and
        len(stride) == len(data_format)):
      self._stride = tuple(stride)[1:-1]
    else:
      self._stride = _fill_and_verify_parameter_shape(stride, self._n, "stride")

    self._rate = _fill_and_verify_parameter_shape(rate, self._n, "rate")

    if any(x > 1 for x in self._stride) and any(x > 1 for x in self._rate):
      raise base.NotSupportedError("Cannot have stride > 1 with rate > 1")

    self._padding = _fill_and_verify_padding(padding, self._n)
    self._conv_op_padding = _padding_to_conv_op_padding(self._padding)

    self._use_bias = use_bias
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = util.check_initializers(
        initializers, self.possible_keys)
    self._partitioners = util.check_partitioners(
        partitioners, self.possible_keys)
    self._regularizers = util.check_regularizers(
        regularizers, self.possible_keys)

    if mask is not None:
      if isinstance(mask, (tf.Tensor, list, tuple, np.ndarray)):
        self._mask = tf.convert_to_tensor(mask)
        if not (tf.float16.is_compatible_with(self._mask.dtype) or
                tf.bfloat16.is_compatible_with(self._mask.dtype) or
                tf.float32.is_compatible_with(self._mask.dtype) or
                tf.float64.is_compatible_with(self._mask.dtype)):
          raise TypeError(
              "Mask needs to have dtype float16, bfloat16, float32 or float64")
        if not self._mask.get_shape().is_fully_defined():
          base.IncompatibleShapeError(
              "Mask needs to have a statically defined shape")
      else:
        raise TypeError("Invalid type for mask: {}".format(type(mask)))
    else:
      self._mask = None

    self._channel_index = _find_channel_index(self._data_format)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

  def _build(self, inputs):
    """Connects the _ConvND module into the graph, with input Tensor `inputs`.

    If this is not the first time the module has been connected to the graph,
    the input Tensor provided here must have the same number of channels, in
    order for the existing variables to be the correct size for the
    multiplication; the batch size and input spatial dimensions may differ for
    each connection.

    Args:
      inputs: A ND Tensor of the same rank as `data_format`, and either of types
      `tf.float16`, `tf.bfloat16`, `tf.float32` or `tf.float64`.

    Returns:
      A ND Tensor of shape [batch_size, output_dim_1, output_dim_2, ...,
          output_channels].

    Raises:
      ValueError: If connecting the module into the graph any time after the
          first time and the inferred size of the input does not match previous
          invocations.
      base.IncompatibleShapeError: If the input tensor has the wrong number
          of dimensions.
      base.UnderspecifiedError: If the channel dimension of `inputs` isn't
          defined.
      base.IncompatibleShapeError: If a mask is present and its shape is
          incompatible with the shape of the weights.
      TypeError: If input Tensor dtype is not compatible with either
          `tf.float16`, `tf.bfloat16`, `tf.float32` or `tf.float64`.
    """
    _verify_inputs(inputs, self._channel_index, self._data_format)
    self._input_shape = tuple(inputs.get_shape().as_list())
    self._input_channels = self._input_shape[self._channel_index]

    self._w = self._construct_w(inputs)

    if self._mask is not None:
      w = self._apply_mask()
    else:
      w = self._w

    inputs = self._pad_input(inputs)
    outputs = self._apply_conv(inputs, w)

    if self._use_bias:
      self._b, outputs = _apply_bias(
          inputs, outputs, self._channel_index, self._data_format,
          self.output_channels, self._initializers, self._partitioners,
          self._regularizers)

    return outputs

  def _pad_input(self, inputs):
    """Pad input in case the desired padding type requires it.

    VALID and SAME padding types are directly supported by tensorflow
    convolution ops, so don't require us to pad input ourselves, at least
    in cases where the same method is used for all dimensions.

    Other padding types (FULL, CAUSAL, REVERSE_CAUSAL) aren't directly supported
    by conv ops but can be implemented by using VALID and padding the input
    appropriately ourselves.

    If different padding types are used for different dimensions, we use VALID
    but pad the input ourselves along any dimensions that require other padding
    types.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.

    Returns:
      inputs: The `inputs` argument that has had any required padding added.
    """
    if all(p == self._conv_op_padding for p in self._padding):
      # All axes require the same padding type that we're going to use for the
      # underlying convolution op, so nothing needs to be done:
      return inputs

    # In all other cases we use VALID as the underlying padding type, and for
    # the axes which require something other than VALID, we pad inputs ourselves
    # before the convolution.
    assert self._conv_op_padding == VALID

    def pad_amount(kernel_size, rate, padding):
      """Pre- and post-padding required for a particular axis before conv op."""
      # The effective kernel size includes any holes/gaps introduced by the
      # dilation rate. It's equal to kernel_size when rate == 1.
      effective_kernel_size = int((kernel_size - 1) * rate + 1)
      if padding == FULL:
        return [effective_kernel_size - 1, effective_kernel_size - 1]
      if padding == CAUSAL:
        return [effective_kernel_size - 1, 0]
      if padding == REVERSE_CAUSAL:
        return [0, effective_kernel_size - 1]
      if padding == SAME:
        return [(effective_kernel_size - 1) // 2, effective_kernel_size // 2]
      # padding == VALID
      return [0, 0]

    paddings = map(pad_amount, self._kernel_shape, self._rate, self._padding)
    if self._data_format.startswith("NC"):  # N, C, ...
      paddings = [[0, 0], [0, 0]] + list(paddings)
    else:  # N, ..., C
      paddings = [[0, 0]] + list(paddings) + [[0, 0]]

    return tf.pad(inputs, paddings)

  def _apply_conv(self, inputs, w):
    """Apply a convolution operation on `inputs` using variable `w`.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.
      w: A weight matrix of the same type as `inputs`.

    Returns:
      outputs: The result of the convolution operation on `inputs`.
    """
    outputs = tf.nn.convolution(inputs, w, strides=self._stride,
                                padding=self._conv_op_padding,
                                dilation_rate=self._rate,
                                data_format=self._data_format)
    return outputs

  def _construct_w(self, inputs):
    """Construct the convolution weight matrix.

    Figures out the shape of the weight matrix, initialize it, and return it.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.

    Returns:
      w: A weight matrix of the same type as `inputs`.
    """
    weight_shape = self._kernel_shape + (self._input_channels,
                                         self.output_channels)

    if "w" not in self._initializers:
      self._initializers["w"] = create_weight_initializer(weight_shape[:-1],
                                                          dtype=inputs.dtype)

    w = tf.get_variable("w",
                        shape=weight_shape,
                        dtype=inputs.dtype,
                        initializer=self._initializers["w"],
                        partitioner=self._partitioners.get("w", None),
                        regularizer=self._regularizers.get("w", None))

    return w

  def _apply_mask(self):
    """Applies the passed-in mask to the convolution matrix.

    Returns:
      w: A copy of the convolution matrix that has had the mask applied.

    Raises:
      base.IncompatibleShapeError: If the mask shape has more dimensions than
          the weight matrix.
      base.IncompatibleShapeError: If the mask and the weight matrix don't
          match on shape.
    """
    w = self._w
    w_shape = w.get_shape()
    mask_shape = self._mask.get_shape()

    if mask_shape.ndims > w_shape.ndims:
      raise base.IncompatibleShapeError(
          "Invalid mask shape: {}. Max shape: {}".format(
              mask_shape.ndims, len(self._data_format)
          )
      )
    if mask_shape != w_shape[:mask_shape.ndims]:
      raise base.IncompatibleShapeError(
          "Invalid mask shape: {}. Weight shape: {}".format(
              mask_shape, w_shape
          )
      )
    # TF broadcasting is a bit fragile.
    # Expand the shape of self._mask by one dim at a time to the right
    # until the rank matches `weight_shape`.
    while self._mask.get_shape().ndims < w_shape.ndims:
      self._mask = tf.expand_dims(self._mask, -1)

    # tf.Variable & tf.ResourceVariable don't support *=.
    w = w * self._mask  # pylint: disable=g-no-augmented-assignment

    return w

  @property
  def output_channels(self):
    """Returns the number of output channels."""
    if callable(self._output_channels):
      self._output_channels = self._output_channels()
    # Channel must be integer.
    self._output_channels = int(self._output_channels)
    return self._output_channels

  @property
  def kernel_shape(self):
    """Returns the kernel shape."""
    return self._kernel_shape

  @property
  def stride(self):
    """Returns the stride."""
    # Backwards compatibility with old stride format.

    return _fill_and_one_pad_stride(self._stride, self._n, self._data_format)

  @property
  def rate(self):
    """Returns the dilation rate."""
    return self._rate

  @property
  def padding(self):
    """Returns the padding algorithm used, if this is the same for all dims.

    Use `.paddings` if you want a tuple with the padding algorithm used for each
    dimension.

    Returns:
      The padding algorithm used, if this is the same for all dimensions.

    Raises:
      ValueError: If different padding algorithms are used for different
        dimensions.
    """
    # This is for backwards compatibility -- previously only a single
    # padding setting was supported across all dimensions.
    if all(p == self._padding[0] for p in self._padding):
      return self._padding[0]
    else:
      raise ValueError("This layer uses different paddings for different "
                       "dimensions. Use .paddings if you want a tuple of "
                       "per-dimension padding settings.")

  @property
  def paddings(self):
    """Returns a tuple with the padding algorithm used for each dimension."""
    return self._padding

  @property
  def conv_op_padding(self):
    """Returns the padding algorithm used for the underlying convolution op."""
    return self._conv_op_padding

  @property
  def w(self):
    """Returns the Variable containing the weight matrix."""
    self._ensure_is_connected()
    return self._w

  @property
  def b(self):
    """Returns the Variable containing the bias.

    Returns:
      Variable object containing the bias, from the most recent __call__.

    Raises:
      base.NotConnectedError: If the module has not been connected to the graph
          yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Conv2D Module when `use_bias=False`.")
    return self._b

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

  @property
  def mask(self):
    """Returns the mask."""
    return self._mask

  @property
  def data_format(self):
    """Returns the data format."""
    return self._data_format

  # Implements Transposable interface.
  @property
  def input_shape(self):
    """Returns the input shape."""
    self._ensure_is_connected()
    return self._input_shape

  @property
  def input_channels(self):
    """Returns the number of input channels."""
    if self._input_channels is None:
      self._ensure_is_connected()
    return self._input_channels

  def clone(self, name=None):
    """Returns a cloned `_ConvND` module.

    Args:
      name: Optional string assigning name of cloned module. The default name
        is constructed by appending "_clone" to `self.module_name`.

    Returns:
      A copy of the current class.
    """
    if name is None:
      name = self.module_name + "_clone"

    return type(self)(output_channels=self.output_channels,
                      kernel_shape=self._kernel_shape,
                      stride=self._stride,
                      rate=self._rate,
                      padding=self._padding,
                      use_bias=self._use_bias,
                      initializers=self._initializers,
                      partitioners=self._partitioners,
                      regularizers=self._regularizers,
                      mask=self._mask,
                      data_format=self._data_format,
                      custom_getter=self._custom_getter,
                      name=name)


class _ConvNDTranspose(base.AbstractModule):
  """Spatial transposed / reverse / up ND convolution module, including bias.

  This acts as a light wrapper around the TensorFlow `conv_nd_transpose` ops,
  abstracting away variable creation and sharing.
  """

  def __init__(self, output_channels, output_shape=None, kernel_shape=None,
               stride=1, padding=SAME, use_bias=True, initializers=None,
               partitioners=None, regularizers=None,
               data_format=DATA_FORMAT_NHWC, custom_getter=None,
               name="conv_nd_transpose"):
    """Constructs a `ConvNDTranspose module`. Support for N = (1, 2, 3).

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/api_guides/python/nn#Convolution

    Args:
      output_channels: Number of output channels.
          Can be either a number or a callable. In the latter case, since the
          function invocation is deferred to graph construction time, the user
          must only ensure `output_channels` can be called, returning an
          integer, when build is called.
      output_shape: Output shape of transpose convolution.
          Can be either an iterable of integers or `Dimension`s, a
          `TensorShape`, or a callable. In the latter case, since the function
          invocation is deferred to graph construction time, the user must only
          ensure that `output_shape` can be called, returning an iterable of
          output shapes when `build` is called. Note that `output_shape` defines
          the size of output signal domain, as opposed to the shape of the
          output `Tensor`. If a None value is given, a default shape is
          automatically calculated (see docstring of
          `_default_transpose_size` function for more details).
      kernel_shape: Sequence of kernel sizes (of size N), or integer that is
          used to define kernel size in all dimensions.
      stride: Sequence of kernel strides (of size N), or integer that is used
          to define stride in all dimensions.
      padding: Padding algorithm, either `snt.SAME` or `snt.VALID`.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b').
      partitioners: Optional dict containing partitioners to partition
          weights (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes
          a single `Tensor` as an input and returns a scalar `Tensor` output,
          e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      data_format: The data format of the input.
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      base.IncompatibleShapeError: If the given kernel shape is neither an
          integer nor a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is neither an integer nor
          a sequence of two or four integers.
      ValueError: If the given padding is not `snt.VALID` or `snt.SAME`.
      ValueError: If the given kernel_shape is `None`.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
    """
    super(_ConvNDTranspose, self).__init__(custom_getter=custom_getter,
                                           name=name)
    self._data_format = data_format
    self._n = len(self._data_format) - 2
    if self._n > 3:
      raise base.NotSupportedError(
          "We only support (1, 2, 3) convolution transpose operations. "
          "Received data format of: {}".format(self._data_format))
    self._output_channels = output_channels

    if output_shape is None:
      self._output_shape = None
      self._use_default_output_shape = True
    else:
      self._use_default_output_shape = False
      if callable(output_shape):
        self._output_shape = output_shape
      else:
        self._output_shape = _fill_and_verify_parameter_shape(output_shape,
                                                              self._n,
                                                              "output_shape")
    if kernel_shape is None:
      raise ValueError("`kernel_shape` cannot be None.")
    self._kernel_shape = _fill_and_verify_parameter_shape(kernel_shape, self._n,
                                                          "kernel")
    if (isinstance(stride, collections.Iterable) and
        len(stride) == len(data_format)):
      if self._data_format.startswith("N") and self._data_format.endswith("C"):
        if not stride[0] == stride[-1] == 1:
          raise base.IncompatibleShapeError(
              "Invalid stride: First and last element must be 1.")
      elif self._data_format.startswith("NC"):
        if not stride[0] == stride[1] == 1:
          raise base.IncompatibleShapeError(
              "Invalid stride: First and second element must be 1.")
      self._stride = tuple(stride)
    else:
      self._stride = _fill_and_one_pad_stride(stride, self._n,
                                              self._data_format)

    self._padding = _verify_conv_op_supported_padding(padding)
    self._use_bias = use_bias
    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = util.check_initializers(
        initializers, self.possible_keys)
    self._partitioners = util.check_partitioners(
        partitioners, self.possible_keys)
    self._regularizers = util.check_regularizers(
        regularizers, self.possible_keys)

    self._channel_index = _find_channel_index(self._data_format)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w", "b"} if use_bias else {"w"}

  def _build(self, inputs):
    """Connects the _ConvNDTranspose module into the graph.

    If this is not the first time the module has been connected to the graph,
    the input Tensor provided here must have the same final N dimensions, in
    order for the existing variables to be the correct size for the
    multiplication. The batch size may differ for each connection.

    Args:
      inputs: A Tensor of shape `data_format` and of type
          `tf.float16`, `tf.bfloat16`, `tf.float32` or `tf.float64`.

    Returns:
      A Tensor of shape `data_format` and of type `tf.float16`, `tf.bfloat16`,
         `tf.float32` or `tf.float64`.

    Raises:
      ValueError: If connecting the module into the graph any time after the
          first time and the inferred size of the input does not match previous
          invocations.
      base.IncompatibleShapeError: If the input tensor has the wrong number
          of dimensions.
      base.UnderspecifiedError: If the channel dimension of `inputs` isn't
          defined.
      base.IncompatibleShapeError: If `output_shape` is an iterable and is not
          in the format `(out_height, out_width)`.
      TypeError: If input Tensor dtype is not compatible with either
          `tf.float16`, `tf.bfloat16`, `tf.float32` or `tf.float64`.
    """
    _verify_inputs(inputs, self._channel_index, self._data_format)
    self._input_shape = tuple(inputs.get_shape().as_list())
    self._input_channels = self._input_shape[self._channel_index]

    # First, figure out what the non-(N,C) dims will be.
    if self._use_default_output_shape:
      def _default_transpose_size_wrapper():
        if self._data_format.startswith("NC"):
          input_size = self._input_shape[2:]
          stride = self.stride[2:]
        else:  # self._data_format == N*WC
          input_size = self._input_shape[1:-1]
          stride = self.stride[1:-1]
        return _default_transpose_size(input_size,
                                       stride,
                                       kernel_shape=self._kernel_shape,
                                       padding=self._padding)

      self._output_shape = _default_transpose_size_wrapper

    if len(self.output_shape) != self._n:
      raise base.IncompatibleShapeError(
          "Output shape must have rank {}, but instead was {}".format(
              self._n, len(self.output_shape)))

    # Now, construct the size of the output, including the N + C dims.
    output_shape = self._infer_all_output_dims(inputs)

    self._w = self._construct_w(inputs)

    if self._n == 1:
      # Add a dimension for the height.
      if self._data_format == DATA_FORMAT_NWC:
        h_dim = 1
        two_dim_conv_data_format = DATA_FORMAT_NHWC
      else:  # self._data_format == DATA_FORMAT_NCW
        h_dim = 2
        two_dim_conv_data_format = DATA_FORMAT_NCHW
      inputs = tf.expand_dims(inputs, h_dim)
      two_dim_conv_stride = self.stride[:h_dim] + (1,) + self.stride[h_dim:]
      outputs = tf.nn.conv2d_transpose(inputs,
                                       self._w,
                                       output_shape,
                                       strides=two_dim_conv_stride,
                                       padding=self._padding,
                                       data_format=two_dim_conv_data_format)
      # Remove the height dimension to return a 3D tensor.
      outputs = tf.squeeze(outputs, [h_dim])
    elif self._n == 2:
      outputs = tf.nn.conv2d_transpose(inputs,
                                       self._w,
                                       output_shape,
                                       strides=self._stride,
                                       padding=self._padding,
                                       data_format=self._data_format)
    else:
      outputs = tf.nn.conv3d_transpose(inputs,
                                       self._w,
                                       output_shape,
                                       strides=self._stride,
                                       padding=self._padding,
                                       data_format=self._data_format)

    if self._use_bias:
      self._b, outputs = _apply_bias(
          inputs, outputs, self._channel_index, self._data_format,
          self._output_channels, self._initializers, self._partitioners,
          self._regularizers)

    outputs = self._recover_shape_information(inputs, outputs)
    return outputs

  def _construct_w(self, inputs):
    """Construct the convolution weight matrix.

    Figures out the shape of the weight matrix, initialize it, and return it.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.

    Returns:
      w: A weight matrix of the same type as `inputs`.
    """
    # Height dim needs to be added to everything for 1D Conv
    # as we'll be using the 2D Conv Transpose op.
    if self._n == 1:
      weight_shape = (1,) + self._kernel_shape + (self.output_channels,
                                                  self._input_channels)
    else:
      weight_shape = self._kernel_shape + (self.output_channels,
                                           self._input_channels)

    if "w" not in self._initializers:
      fan_in_shape = self._kernel_shape + (self._input_channels,)
      self._initializers["w"] = create_weight_initializer(fan_in_shape,
                                                          dtype=inputs.dtype)
    w = tf.get_variable("w",
                        shape=weight_shape,
                        dtype=inputs.dtype,
                        initializer=self._initializers["w"],
                        partitioner=self._partitioners.get("w", None),
                        regularizer=self._regularizers.get("w", None))
    return w

  def _infer_all_output_dims(self, inputs):
    """Calculate the output shape for `inputs` after a deconvolution.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.

    Returns:
      output_shape: A tensor of shape (`batch_size`, `conv_output_shape`).
    """
    # Use tensorflow shape op to manipulate inputs shape, so that unknown batch
    # size - which can happen when using input placeholders - is handled
    # correcly.
    batch_size = tf.expand_dims(tf.shape(inputs)[0], 0)
    out_channels = (self.output_channels,)

    # Height dim needs to be added to everything for 1D Conv
    # as we'll be using the 2D Conv Transpose op.
    if self._n == 1:
      out_shape = (1,) + self.output_shape
    else:
      out_shape = self.output_shape

    if self._data_format.startswith("NC"):
      out_shape_tuple = out_channels + out_shape
    elif self._data_format.startswith("N") and self._data_format.endswith("C"):
      out_shape_tuple = out_shape + out_channels

    output_shape = tf.concat([batch_size, out_shape_tuple], 0)
    return output_shape

  def _recover_shape_information(self, inputs, outputs):
    """Recover output tensor shape value to enable shape inference.

    The batch size of `inputs` isn't preserved by the convolution op. Calculate
    what the proper output shape will be for `outputs`.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.
      outputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`. The output of `inputs`
          from a transpose convolution op.

    Returns:
      outputs: The passed-in `outputs` with all shape information filled in.
    """
    batch_size_value = inputs.get_shape()[0]
    if self._data_format.startswith("NC"):
      output_shape_value = ((batch_size_value, self.output_channels) +
                            self.output_shape)
    elif self._data_format.startswith("N") and self._data_format.endswith("C"):
      output_shape_value = ((batch_size_value,) + self.output_shape +
                            (self.output_channels,))
    outputs.set_shape(output_shape_value)
    return outputs

  @property
  def output_channels(self):
    """Returns the number of output channels."""
    if callable(self._output_channels):
      self._output_channels = self._output_channels()
    # Channel must be integer.
    self._output_channels = int(self._output_channels)
    return self._output_channels

  @property
  def kernel_shape(self):
    """Returns the kernel shape."""
    return self._kernel_shape

  @property
  def stride(self):
    """Returns the stride."""
    return self._stride

  @property
  def output_shape(self):
    """Returns the output shape."""
    if self._output_shape is None:
      self._ensure_is_connected()
    if callable(self._output_shape):
      self._output_shape = tuple(self._output_shape())
    return self._output_shape

  @property
  def padding(self):
    """Returns the padding algorithm."""
    return self._padding

  @property
  def conv_op_padding(self):
    """Returns the padding algorithm used for the underlying convolution op."""
    return self._padding

  @property
  def w(self):
    """Returns the Variable containing the weight matrix."""
    self._ensure_is_connected()
    return self._w

  @property
  def b(self):
    """Returns the Variable containing the bias.

    Returns:
      Variable object containing the bias, from the most recent __call__.

    Raises:
      base.NotConnectedError: If the module has not been connected to the graph
          yet, meaning the variables do not exist.
      AttributeError: If the module does not use bias.
    """
    self._ensure_is_connected()
    if not self._use_bias:
      raise AttributeError(
          "No bias Variable in Conv2DTranspose Module when `use_bias=False`.")
    return self._b

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

  @property
  def input_shape(self):
    """Returns the input shape."""
    self._ensure_is_connected()
    return self._input_shape

  @property
  def input_channels(self):
    """Returns the number of input channels."""
    self._ensure_is_connected()
    return self._input_channels


class Conv1D(_ConvND, base.Transposable):
  """1D convolution module, including optional bias.

  This acts as a light wrapper around the class `_ConvND`.
  """

  def __init__(self, output_channels, kernel_shape, stride=1, rate=1,
               padding=SAME, use_bias=True, initializers=None,
               partitioners=None, regularizers=None, mask=None,
               data_format=DATA_FORMAT_NWC, custom_getter=None,
               name="conv_1d"):
    """Constructs a Conv1D module.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/api_guides/python/nn#Convolution

    Args:
      output_channels: Number of output channels. `output_channels` can be
          either a number or a callable. In the latter case, since the function
          invocation is deferred to graph construction time, the user must only
          ensure that output_channels can be called, returning an integer,
          when `build` is called.
      kernel_shape: Sequence of kernel sizes (of size 1), or integer that is
          used to define kernel size in all dimensions.
      stride: Sequence of kernel strides (of size 1), or integer that is used to
          define stride in all dimensions.
      rate: Sequence of dilation rates (of size 1), or integer that is used to
          define dilation rate in all dimensions. 1 corresponds to standard
          convolution, `rate > 1` corresponds to dilated convolution. Cannot be
          > 1 if any of `stride` is also > 1.
      padding: Padding algorithm. Either `snt.SAME`, `snt.VALID`, `snt.FULL`,
          `snt.CAUSAL`, `snt.REVERSE_CAUSAL`, or a sequence of these paddings
          of length 1.
          * snt.SAME and snt.VALID are explained in the Tensorflow docs at
            https://www.tensorflow.org/api_guides/python/nn#Convolution.
          * snt.FULL pre- and post-pads with the maximum padding which does not
            result in a convolution over just padded elements.
          * snt.CAUSAL pre-pads to ensure that each output value only depends on
            input values at the same or preceding indices ("no dependence on the
            future").
          * snt.REVERSE_CAUSAL post-pads to ensure that each output value only
            depends on input values at the same or *greater* indices ("no
            dependence on the past").
          If you use the same padding for all dimensions, and it is one of SAME
          or VALID, then this is supported directly by the underlying
          convolution op. In all other cases, the input data will be padded
          using tf.pad before calling the convolution op.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializer for the
          weights is a truncated normal initializer, which is commonly used
          when the inputs are zero centered (see
          https://arxiv.org/pdf/1502.03167v3.pdf). The default initializer for
          the bias is a zero initializer.
      partitioners: Optional dict containing partitioners to partition
          weights (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes
          a single `Tensor` as an input and returns a scalar `Tensor` output,
          e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      mask: A convertible to a 3D tensor which is multiplied
          component-wise with the weights (Optional).
      data_format: A string. Specifies whether the channel dimension
          of the input and output is the last dimension (default, NWC), or the
          second dimension (NCW).
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      base.IncompatibleShapeError: If the given kernel shape is not an integer;
          or if the given kernel shape is not a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is not an integer; or if
          the given stride is not a sequence of two integers.
      base.IncompatibleShapeError: If the given rate is not an integer; or if
          the given rate is not a sequence of two integers.
      base.IncompatibleShapeError: If a mask is a TensorFlow Tensor with
          a not fully defined shape.
      base.NotSupportedError: If rate in any dimension and the stride in any
          dimension are simultaneously > 1.
      ValueError: If the given padding is not `snt.VALID` or `snt.SAME`.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      TypeError: If mask is given and it is not convertible to a Tensor.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
      ValueError: If the given data_format is not a supported format (see
          `SUPPORTED_1D_DATA_FORMATS`).
    """
    if data_format not in SUPPORTED_1D_DATA_FORMATS:
      raise ValueError("Invalid data_format {:s}. Allowed formats "
                       "{}".format(data_format, SUPPORTED_1D_DATA_FORMATS))
    super(Conv1D, self).__init__(
        output_channels=output_channels, kernel_shape=kernel_shape,
        stride=stride, rate=rate, padding=padding, use_bias=use_bias,
        initializers=initializers, partitioners=partitioners,
        regularizers=regularizers, mask=mask, data_format=data_format,
        custom_getter=custom_getter, name=name)

  # Implement Transposable interface
  def transpose(self, name=None):
    """Returns matching `Conv1DTranspose` module.

    Args:
      name: Optional string assigning name of transpose module. The default name
          is constructed by appending "_transpose" to `self.name`.

    Returns:
      `Conv1DTranspose` module.

    Raises:
     base.NotSupportedError: If `rate` in any dimension > 1.
    """
    if any(x > 1 for x in self._rate):
      raise base.NotSupportedError(
          "Cannot transpose a dilated convolution module.")

    if any(p != self._conv_op_padding for p in self._padding):
      raise base.NotSupportedError(
          "Cannot tranpose a convolution using mixed paddings or paddings "
          "other than SAME or VALID.")

    def output_shape():
      if self._data_format == DATA_FORMAT_NCW:
        return (self._input_shape[2],)
      else:  # data_format = DATA_FORMAT_NWC
        return (self._input_shape[1],)

    if name is None:
      name = self.module_name + "_transpose"
    return Conv1DTranspose(output_channels=lambda: self._input_channels,
                           output_shape=output_shape,
                           kernel_shape=self._kernel_shape,
                           stride=self._stride,
                           padding=self._conv_op_padding,
                           use_bias=self._use_bias,
                           initializers=self._initializers,
                           partitioners=self._partitioners,
                           regularizers=self._regularizers,
                           data_format=self._data_format,
                           custom_getter=self._custom_getter,
                           name=name)


class Conv1DTranspose(_ConvNDTranspose, base.Transposable):
  """1D transposed / reverse / up 1D convolution module, including bias.

  This performs a 1D transpose convolution by lightly wrapping the TensorFlow op
  `tf.nn.conv2d_transpose`, setting the size of the height dimension of the
  image to 1.
  """

  def __init__(self, output_channels, output_shape=None, kernel_shape=None,
               stride=1, padding=SAME, use_bias=True, initializers=None,
               partitioners=None, regularizers=None,
               data_format=DATA_FORMAT_NWC, custom_getter=None,
               name="conv_1d_transpose"):
    """Constructs a Conv1DTranspose module.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/api_guides/python/nn#Convolution

    Args:
      output_channels: Number of output channels. Can be either a number or a
          callable. In the latter case, since the function invocation is
          deferred to graph construction time, the user must only ensure
          `output_channels` can be called, returning an integer, when build is
          called.
      output_shape: Output shape of transpose convolution. Can be either a
          number or a callable. In the latter case, since the function
          invocation is deferred to graph construction time, the user must only
          ensure that `output_shape` can be called, returning an iterable of
          format `(out_length)` when build is called. If a None
          value is given, a default shape is automatically calculated (see
          docstring of _default_transpose_size function for more details).
      kernel_shape: Sequence of kernel sizes (of size 1), or integer that is
          used to define kernel size in all dimensions.
      stride: Sequence of kernel strides (of size 1), or integer that is used to
          define stride in all dimensions.
      padding: Padding algorithm, either `snt.SAME` or `snt.VALID`.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b').
      partitioners: Optional dict containing partitioners to partition
          weights (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes
          a single `Tensor` as an input and returns a scalar `Tensor` output,
          e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      data_format: A string. Specifies whether the channel dimension
          of the input and output is the last dimension (default, NWC), or the
          second dimension (NCW).
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      base.IncompatibleShapeError: If the given kernel shape is not an integer;
          or if the given kernel shape is not a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is not an integer; or if
          the given stride is not a sequence of two or four integers.
      ValueError: If the given padding is not `snt.VALID` or `snt.SAME`.
      ValueError: If the given kernel_shape is `None`.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
      ValueError: If the given data_format is not a supported format (see
          `SUPPORTED_1D_DATA_FORMATS`).
    """
    if data_format not in SUPPORTED_1D_DATA_FORMATS:
      raise ValueError("Invalid data_format {:s}. Allowed formats "
                       "{}".format(data_format, SUPPORTED_1D_DATA_FORMATS))

    super(Conv1DTranspose, self).__init__(
        output_channels=output_channels, output_shape=output_shape,
        kernel_shape=kernel_shape, stride=stride, padding=padding,
        use_bias=use_bias, initializers=initializers,
        partitioners=partitioners, regularizers=regularizers,
        data_format=data_format, custom_getter=custom_getter, name=name
    )

  # Implement Transposable interface.
  def transpose(self, name=None):
    """Returns matching `Conv1D` module.

    Args:
      name: Optional string assigning name of transpose module. The default name
        is constructed by appending "_transpose" to `self.name`.

    Returns:
      `Conv1D` module.
    """
    if name is None:
      name = self.module_name + "_transpose"

    if self._data_format == DATA_FORMAT_NWC:
      stride = self._stride[1:-1]
    else:  # self._data_format == DATA_FORMAT_NCW
      stride = self._stride[2:]

    return Conv1D(output_channels=lambda: self.input_channels,
                  kernel_shape=self.kernel_shape,
                  stride=stride,
                  padding=self.padding,
                  use_bias=self._use_bias,
                  initializers=self.initializers,
                  partitioners=self.partitioners,
                  regularizers=self.regularizers,
                  data_format=self._data_format,
                  custom_getter=self._custom_getter,
                  name=name)


class CausalConv1D(_ConvND):
  """1D convolution module, including optional bias.

  This is deprecated, please use the padding=CAUSAL argument to Conv1D.

  This acts as a light wrapper around _ConvND ensuring that the outputs at index
  `i` only depend on indices smaller than `i` (also known as a causal
  convolution). For further details on the theoretical background, refer to:

  https://arxiv.org/abs/1610.10099
  """

  def __init__(self, output_channels, kernel_shape,
               stride=1, rate=1, use_bias=True, initializers=None,
               partitioners=None, regularizers=None, mask=None,
               padding=CAUSAL, data_format=DATA_FORMAT_NWC,
               custom_getter=None, name="causal_conv_1d"):
    """Constructs a CausalConv1D module.

    This is deprecated, please use the padding=CAUSAL argument to Conv1D.

    Args:
      output_channels: Number of output channels. `output_channels` can be
          either a number or a callable. In the latter case, since the function
          invocation is deferred to graph construction time, the user must only
          ensure that output_channels can be called, returning an integer,
          when `build` is called.
      kernel_shape: Sequence of kernel sizes (of size 1), or integer that is
          used to define kernel size in all dimensions.
      stride: Sequence of kernel strides (of size 1), or integer that is used to
          define stride in all dimensions.
      rate: Sequence of dilation rates (of size 1), or integer that is used to
          define dilation rate in all dimensions. 1 corresponds to standard
          convolution, `rate > 1` corresponds to dilated convolution. Cannot be
          > 1 if any of `stride` is also > 1.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializer for the
          weights is a truncated normal initializer, which is commonly used
          when the inputs are zero centered (see
          https://arxiv.org/pdf/1502.03167v3.pdf). The default initializer for
          the bias is a zero initializer.
      partitioners: Optional dict containing partitioners to partition
          weights (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes
          a single `Tensor` as an input and returns a scalar `Tensor` output,
          e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      mask: A convertible to a 3D tensor which is multiplied
          component-wise with the weights (Optional).
      padding: Padding algorithm. Should be `snt.CAUSAL`.
      data_format: A string. Specifies whether the channel dimension
          of the input and output is the last dimension (default, NWC), or the
          second dimension (NCW).
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      base.IncompatibleShapeError: If the given kernel shape is not an integer;
          or if the given kernel shape is not a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is not an integer; or if
          the given stride is not a sequence of two integers.
      base.IncompatibleShapeError: If the given rate is not an integer; or if
          the given rate is not a sequence of two integers.
      base.IncompatibleShapeError: If a mask is a TensorFlow Tensor with
          a not fully defined shape.
      base.NotSupportedError: If rate in any dimension and the stride in any
          dimension are simultaneously > 1.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      TypeError: If mask is given and it is not convertible to a Tensor.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
      ValueError: If the given data_format is not a supported format (see
          `SUPPORTED_1D_DATA_FORMATS`).
    """
    util.deprecation_warning(
        "CausalConv1D is deprecated, please use Conv1D with padding=CAUSAL.")
    if data_format not in SUPPORTED_1D_DATA_FORMATS:
      raise ValueError("Invalid data_format {:s}. Allowed formats "
                       "{}".format(data_format, SUPPORTED_1D_DATA_FORMATS))
    if padding != CAUSAL:
      # This used to be required to be VALID, which is now rather ambiguous.
      # Supporting VALID for now but with a warning:
      util.deprecation_warning(
          "You specified a non-casual padding type for CausalConv1D, this has "
          "been ignored and you will get CAUSAL padding. Note CausalConv1D is "
          "deprecated, please switch to Conv1D with padding=CAUSAL.")
    super(CausalConv1D, self).__init__(
        output_channels=output_channels, kernel_shape=kernel_shape,
        stride=stride, rate=rate, padding=CAUSAL, use_bias=use_bias,
        initializers=initializers, partitioners=partitioners,
        regularizers=regularizers, mask=mask,
        data_format=data_format,
        custom_getter=custom_getter, name=name)


class Conv2D(_ConvND, base.Transposable):
  """Spatial convolution and dilated convolution module, including bias.

  This acts as a light wrapper around the class `_ConvND`.
  """

  def __init__(self, output_channels, kernel_shape, stride=1, rate=1,
               padding=SAME, use_bias=True, initializers=None,
               partitioners=None, regularizers=None, mask=None,
               data_format=DATA_FORMAT_NHWC, custom_getter=None,
               name="conv_2d"):
    """Constructs a Conv2D module.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/api_guides/python/nn#Convolution

    Args:
      output_channels: Number of output channels. `output_channels` can be
          either a number or a callable. In the latter case, since the function
          invocation is deferred to graph construction time, the user must only
          ensure that output_channels can be called, returning an integer,
          when `build` is called.
      kernel_shape: Sequence of kernel sizes (of size 2), or integer that is
          used to define kernel size in all dimensions.
      stride: Sequence of kernel strides (of size 2), or integer that is used to
          define stride in all dimensions.
      rate: Sequence of dilation rates (of size 2), or integer that is used to
          define dilation rate in all dimensions. 1 corresponds to standard 2D
          convolution, `rate > 1` corresponds to dilated convolution. Cannot be
          > 1 if any of `stride` is also > 1.
      padding: Padding algorithm. Either `snt.SAME`, `snt.VALID`, `snt.FULL`,
          `snt.CAUSAL`, `snt.REVERSE_CAUSAL`, or a sequence of these paddings
          of length 2.
          * snt.SAME and snt.VALID are explained in the Tensorflow docs at
            https://www.tensorflow.org/api_guides/python/nn#Convolution.
          * snt.FULL pre- and post-pads with the maximum padding which does not
            result in a convolution over just padded elements.
          * snt.CAUSAL pre-pads to ensure that each output value only depends on
            input values at the same or preceding indices ("no dependence on the
            future").
          * snt.REVERSE_CAUSAL post-pads to ensure that each output value only
            depends on input values at the same or *greater* indices ("no
            dependence on the past").
          If you use the same padding for all dimensions, and it is one of SAME
          or VALID, then this is supported directly by the underlying
          convolution op. In all other cases, the input data will be padded
          using tf.pad before calling the convolution op.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializer for the
          weights is a truncated normal initializer, which is commonly used
          when the inputs are zero centered (see
          https://arxiv.org/pdf/1502.03167v3.pdf). The default initializer for
          the bias is a zero initializer.
      partitioners: Optional dict containing partitioners to partition
          weights (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes
          a single `Tensor` as an input and returns a scalar `Tensor` output,
          e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      mask: A convertible to a 4D tensor which is multiplied
          component-wise with the weights (Optional).
      data_format: A string. Specifies whether the channel dimension
          of the input and output is the last dimension (default, NHWC), or the
          second dimension (NCHW).
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      base.IncompatibleShapeError: If the given kernel shape is not an integer;
          or if the given kernel shape is not a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is not an integer; or if
          the given stride is not a sequence of two integers.
      base.IncompatibleShapeError: If the given rate is not an integer; or if
          the given rate is not a sequence of two integers.
      base.IncompatibleShapeError: If a mask is given and its rank is neither 2
          nor 4, or if it is a TensorFlow Tensor with a not fully defined shape.
      base.NotSupportedError: If rate in any dimension and the stride in any
          dimension are simultaneously > 1.
      ValueError: If the given padding is not `snt.VALID`, `snt.SAME`,
          `snt.FULL`, `snt.CAUSAL`, `snt.REVERSE_CAUSAL` or a sequence of these.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      TypeError: If mask is given and it is not convertible to a Tensor.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
      ValueError: If the given data_format is not a supported format (see
        `SUPPORTED_2D_DATA_FORMATS`).
    """
    if data_format not in SUPPORTED_2D_DATA_FORMATS:
      raise ValueError("Invalid data_format {:s}. Allowed formats "
                       "{}".format(data_format, SUPPORTED_2D_DATA_FORMATS))
    super(Conv2D, self).__init__(
        output_channels=output_channels, kernel_shape=kernel_shape,
        stride=stride, rate=rate, padding=padding, use_bias=use_bias,
        initializers=initializers, partitioners=partitioners,
        regularizers=regularizers, mask=mask, data_format=data_format,
        custom_getter=custom_getter, name=name)

  # Implements Transposable interface.
  def transpose(self, name=None):
    """Returns matching `Conv2DTranspose` module.

    Args:
      name: Optional string assigning name of transpose module. The default name
        is constructed by appending "_transpose" to `self.name`.

    Returns:
      `Conv2DTranspose` module.

    Raises:
     base.NotSupportedError: If `rate` in any dimension > 1.
    """
    if any(x > 1 for x in self._rate):
      raise base.NotSupportedError(
          "Cannot transpose a dilated convolution module.")

    if any(p != self._conv_op_padding for p in self._padding):
      raise base.NotSupportedError(
          "Cannot tranpose a convolution using mixed paddings or paddings "
          "other than SAME or VALID.")

    if name is None:
      name = self.module_name + "_transpose"

    def output_shape():
      if self._data_format == DATA_FORMAT_NCHW:
        return self.input_shape[2:4]
      else:  # data_format == DATA_FORMAT_NHWC
        return self.input_shape[1:3]

    return Conv2DTranspose(output_channels=lambda: self._input_channels,
                           output_shape=output_shape,
                           kernel_shape=self._kernel_shape,
                           stride=self._stride,
                           padding=self._conv_op_padding,
                           use_bias=self._use_bias,
                           initializers=self._initializers,
                           partitioners=self._partitioners,
                           regularizers=self._regularizers,
                           data_format=self._data_format,
                           custom_getter=self._custom_getter,
                           name=name)


class Conv2DTranspose(_ConvNDTranspose, base.Transposable):
  """Spatial transposed / reverse / up 2D convolution module, including bias.

  This acts as a light wrapper around the TensorFlow op `tf.nn.conv2d_transpose`
  abstracting away variable creation and sharing.
  """

  def __init__(self, output_channels, output_shape=None, kernel_shape=None,
               stride=1, padding=SAME, use_bias=True, initializers=None,
               partitioners=None, regularizers=None,
               data_format=DATA_FORMAT_NHWC, custom_getter=None,
               name="conv_2d_transpose"):
    """Constructs a `Conv2DTranspose module`.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/api_guides/python/nn#Convolution

    Args:
      output_channels: Number of output channels.
          Can be either a number or a callable. In the latter case, since the
          function invocation is deferred to graph construction time, the user
          must only ensure `output_channels` can be called, returning an
          integer, when build is called.
      output_shape: Output shape of transpose convolution.
          Can be either an iterable of integers or a callable. In the latter
          case, since the function invocation is deferred to graph construction
          time, the user must only ensure that `output_shape` can be called,
          returning an iterable of format `(out_height, out_width)` when `build`
          is called. Note that `output_shape` defines the size of output signal
          domain, as opposed to the shape of the output `Tensor`. If a None
          value is given, a default shape is automatically calculated (see
          docstring of _default_transpose_size function for more details).
      kernel_shape: Sequence of kernel sizes (of size 2), or integer that is
          used to define kernel size in all dimensions.
      stride: Sequence of kernel strides (of size 2), or integer that is used to
          define stride in all dimensions.
      padding: Padding algorithm, either `snt.SAME` or `snt.VALID`.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b').
      partitioners: Optional dict containing partitioners to partition
          weights (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes
          a single `Tensor` as an input and returns a scalar `Tensor` output,
          e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      data_format: A string. Specifies whether the channel dimension
          of the input and output is the last dimension (default, NHWC), or the
          second dimension ("NCHW").
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the`
          tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      base.IncompatibleShapeError: If the given kernel shape is neither an
          integer nor a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is neither an integer nor
          a sequence of two or four integers.
      ValueError: If the given padding is not `snt.VALID` or `snt.SAME`.
      ValueError: If the given kernel_shape is `None`.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
      ValueError: If the given data_format is not a supported format (see
          `SUPPORTED_2D_DATA_FORMATS`).
    """
    if data_format not in SUPPORTED_2D_DATA_FORMATS:
      raise ValueError("Invalid data_format {:s}. Allowed formats "
                       "{}".format(data_format, SUPPORTED_2D_DATA_FORMATS))

    super(Conv2DTranspose, self).__init__(
        output_channels=output_channels, output_shape=output_shape,
        kernel_shape=kernel_shape, stride=stride, padding=padding,
        use_bias=use_bias, initializers=initializers,
        partitioners=partitioners, regularizers=regularizers,
        data_format=data_format, custom_getter=custom_getter, name=name
    )

  # Implements Transposable interface.
  def transpose(self, name=None):
    """Returns matching `Conv2D` module.

    Args:
      name: Optional string assigning name of transpose module. The default name
          is constructed by appending "_transpose" to `self.name`.

    Returns:
      `Conv2D` module.
    """
    if name is None:
      name = self.module_name + "_transpose"

    if self._data_format == DATA_FORMAT_NHWC:
      stride = self._stride[1:-1]
    else:  # self._data_format == DATA_FORMAT_NCHW
      stride = self._stride[2:]

    return Conv2D(output_channels=lambda: self.input_channels,
                  kernel_shape=self._kernel_shape,
                  stride=stride,
                  padding=self._padding,
                  use_bias=self._use_bias,
                  initializers=self._initializers,
                  partitioners=self._partitioners,
                  regularizers=self._regularizers,
                  data_format=self._data_format,
                  custom_getter=self._custom_getter,
                  name=name)


class Conv3D(_ConvND, base.Transposable):
  """Volumetric convolution module, including optional bias.

  This acts as a light wrapper around the class `_ConvND`.
  """

  def __init__(self, output_channels, kernel_shape, stride=1, rate=1,
               padding=SAME, use_bias=True, initializers=None,
               partitioners=None, regularizers=None, mask=None,
               data_format=DATA_FORMAT_NDHWC, custom_getter=None,
               name="conv_3d"):
    """Constructs a Conv3D module.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/api_guides/python/nn#Convolution

    Args:
      output_channels: Number of output channels. `output_channels` can be
          either a number or a callable. In the latter case, since the function
          invocation is deferred to graph construction time, the user must only
          ensure that output_channels can be called, returning an integer,
          when `build` is called.
      kernel_shape: Sequence of kernel sizes (of size 3), or integer that is
          used to define kernel size in all dimensions.
      stride: Sequence of kernel strides (of size 3), or integer that is used to
          define stride in all dimensions.
      rate: Sequence of dilation rates (of size 3), or integer that is used to
          define dilation rate in all dimensions. 1 corresponds to standard 3D
          convolution, `rate > 1` corresponds to dilated convolution. Cannot be
          > 1 if any of `stride` is also > 1.
      padding: Padding algorithm. Either `snt.SAME`, `snt.VALID`, `snt.FULL`,
          `snt.CAUSAL`, `snt.REVERSE_CAUSAL`, or a sequence of these paddings
          of length 3.
          * snt.SAME and snt.VALID are explained in the Tensorflow docs at
            https://www.tensorflow.org/api_guides/python/nn#Convolution.
          * snt.FULL pre- and post-pads with the maximum padding which does not
            result in a convolution over just padded elements.
          * snt.CAUSAL pre-pads to ensure that each output value only depends on
            input values at the same or preceding indices ("no dependence on the
            future").
          * snt.REVERSE_CAUSAL post-pads to ensure that each output value only
            depends on input values at the same or *greater* indices ("no
            dependence on the past").
          If you use the same padding for all dimensions, and it is one of SAME
          or VALID, then this is supported directly by the underlying
          convolution op. In all other cases, the input data will be padded
          using tf.pad before calling the convolution op.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializer for the
          weights is a truncated normal initializer, which is commonly used
          when the inputs are zero centered (see
          https://arxiv.org/pdf/1502.03167v3.pdf). The default initializer for
          the bias is a zero initializer.
      partitioners: Optional dict containing partitioners to partition
          weights (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes
          a single `Tensor` as an input and returns a scalar `Tensor` output,
          e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      mask: An object convertible to a 5D tensor which is multiplied
          component-wise with the weights (Optional).
      data_format: A string. Specifies whether the channel dimension
          of the input and output is the last dimension (default, NDHWC), or
          the second dimension (NCDHW).
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      base.IncompatibleShapeError: If the given kernel shape is not an integer;
          or if the given kernel shape is not a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is not an integer; or if
          the given stride is not a sequence of two or four integers.
      base.IncompatibleShapeError: If the given rate is not an integer; or if
          the given rate is not a sequence of two integers.
      base.NotSupportedError: If rate in any dimension and the stride in any
          dimension are simultaneously > 1.
      ValueError: If the given padding is not `snt.VALID`, `snt.SAME`,
          `snt.FULL`, `snt.CAUSAL`, `snt.REVERSE_CAUSAL` or a sequence of these.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
      ValueError: If the given data_format is not a supported format (see
          `SUPPORTED_3D_DATA_FORMATS`).
    """
    if data_format not in SUPPORTED_3D_DATA_FORMATS:
      raise ValueError("Invalid data_format {:s}. Allowed formats "
                       "{}".format(data_format, SUPPORTED_3D_DATA_FORMATS))
    super(Conv3D, self).__init__(
        output_channels=output_channels, kernel_shape=kernel_shape,
        stride=stride, rate=rate, padding=padding, use_bias=use_bias,
        initializers=initializers, partitioners=partitioners,
        regularizers=regularizers, mask=mask, data_format=data_format,
        custom_getter=custom_getter, name=name)

  # Implements Transposable interface.
  def transpose(self, name=None):
    """Returns matching `Conv3DTranspose` module.

    Args:
      name: Optional string assigning name of transpose module. The default name
        is constructed by appending "_transpose" to `self.name`.

    Returns:
      `Conv3DTranspose` module.

    Raises:
     base.NotSupportedError: If `rate` in any dimension > 1.
    """
    if any(x > 1 for x in self._rate):
      raise base.NotSupportedError(
          "Cannot transpose a dilated convolution module.")

    if any(p != self._conv_op_padding for p in self._padding):
      raise base.NotSupportedError(
          "Cannot tranpose a convolution using mixed paddings or paddings "
          "other than SAME or VALID.")

    def output_shape():
      if self._data_format == DATA_FORMAT_NCDHW:
        return self.input_shape[2:]
      else:  # data_format == DATA_FORMAT_NDHWC
        return self.input_shape[1:4]

    if name is None:
      name = self.module_name + "_transpose"
    return Conv3DTranspose(output_channels=lambda: self._input_channels,
                           output_shape=output_shape,
                           kernel_shape=self._kernel_shape,
                           stride=self._stride,
                           padding=self._conv_op_padding,
                           use_bias=self._use_bias,
                           initializers=self._initializers,
                           partitioners=self._partitioners,
                           regularizers=self._regularizers,
                           data_format=self._data_format,
                           custom_getter=self._custom_getter,
                           name=name)


class Conv3DTranspose(_ConvNDTranspose, base.Transposable):
  """Volumetric transposed / reverse / up 3D convolution module, including bias.

  This acts as a light wrapper around the TensorFlow op `tf.nn.conv3d_transpose`
  abstracting away variable creation and sharing.
  """

  def __init__(self, output_channels, output_shape=None, kernel_shape=None,
               stride=1, padding=SAME, use_bias=True, initializers=None,
               partitioners=None, regularizers=None,
               data_format=DATA_FORMAT_NDHWC, custom_getter=None,
               name="conv_3d_transpose"):
    """Constructs a `Conv3DTranspose` module.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/api_guides/python/nn#Convolution

    Args:
      output_channels: Number of output channels. `output_channels` can be
          either a number or a callable. In the latter case, since the function
          invocation is deferred to graph construction time, the user must only
          ensure `output_channels` can be called, returning an integer, when
          `build` is called.
      output_shape: Output shape of transpose convolution.
          Can be either an iterable of integers or a callable. In the latter
          case, since the function invocation is deferred to graph construction
          time, the user must only ensure that `output_shape` can be called,
          returning an iterable of format `(out_depth, out_height, out_width)`
          when `build` is called. Note that `output_shape` defines the size of
          output signal domain, as opposed to the shape of the output `Tensor`.
          If a None value is given, a default shape is automatically calculated
          (see docstring of _default_transpose_size function for more details).
      kernel_shape: Sequence of kernel sizes (of size 3), or integer that is
          used to define kernel size in all dimensions.
      stride: Sequence of kernel strides (of size 3), or integer that is used to
          define stride in all dimensions.
      padding: Padding algorithm, either `snt.SAME` or `snt.VALID`.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b').
      partitioners: Optional dict containing partitioners to partition
          weights (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes
          a single `Tensor` as an input and returns a scalar `Tensor` output,
          e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      data_format: A string. Specifies whether the channel dimension
          of the input and output is the last dimension (default, NDHWC), or the
          second dimension (NCDHW).
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      module.IncompatibleShapeError: If the given kernel shape is neither an
          integer nor a sequence of three integers.
      module.IncompatibleShapeError: If the given stride is neither an integer
          nor a sequence of three or five integers.
      ValueError: If the given padding is not `snt.VALID` or `snt.SAME`.
      ValueError: If the given kernel_shape is `None`.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
      ValueError: If the given data_format is not a supported format (see
          `SUPPORTED_3D_DATA_FORMATS`).
    """
    if data_format not in SUPPORTED_3D_DATA_FORMATS:
      raise ValueError("Invalid data_format {:s}. Allowed formats "
                       "{}".format(data_format, SUPPORTED_3D_DATA_FORMATS))

    super(Conv3DTranspose, self).__init__(
        output_channels=output_channels, output_shape=output_shape,
        kernel_shape=kernel_shape, stride=stride, padding=padding,
        use_bias=use_bias, initializers=initializers,
        partitioners=partitioners, regularizers=regularizers,
        data_format=data_format, custom_getter=custom_getter, name=name
    )

  # Implement Transposable interface
  def transpose(self, name=None):
    """Returns transposed Conv3DTranspose module, i.e. a Conv3D module."""
    if name is None:
      name = self.module_name + "_transpose"

    if self._data_format == DATA_FORMAT_NDHWC:
      stride = self._stride[1:-1]
    else:  # self._data_format == DATA_FORMAT_NCDHW
      stride = self._stride[2:]

    return Conv3D(output_channels=lambda: self.input_channels,
                  kernel_shape=self._kernel_shape,
                  stride=stride,
                  padding=self._padding,
                  use_bias=self._use_bias,
                  initializers=self._initializers,
                  partitioners=self._partitioners,
                  regularizers=self._regularizers,
                  data_format=self._data_format,
                  custom_getter=self._custom_getter,
                  name=name)


class InPlaneConv2D(_ConvND):
  """Applies an in-plane convolution to each channel with tied filter weights.

  This acts as a light wrapper around the TensorFlow op
  `tf.nn.depthwise_conv2d`; it differs from the DepthWiseConv2D module in that
  it has tied weights (i.e. the same filter) for all the in-out channel pairs.
  """

  def __init__(self, kernel_shape, stride=1, padding=SAME, use_bias=True,
               initializers=None, partitioners=None, regularizers=None,
               data_format=DATA_FORMAT_NHWC, custom_getter=None,
               name="in_plane_conv2d"):
    """Constructs an InPlaneConv2D module.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/api_guides/python/nn#Convolution

    Args:
      kernel_shape: Iterable with 2 elements in the layout [filter_height,
          filter_width]; or integer that is used to define the list in all
          dimensions.
      stride: Iterable with 2 or 4 elements of kernel strides, or integer that
          is used to define stride in all dimensions.
      padding: Padding algorithm. Either `snt.SAME`, `snt.VALID`, `snt.FULL`,
          `snt.CAUSAL`, `snt.REVERSE_CAUSAL`, or a sequence of these paddings
          of length 2.
          * snt.SAME and snt.VALID are explained in the Tensorflow docs at
            https://www.tensorflow.org/api_guides/python/nn#Convolution.
          * snt.FULL pre- and post-pads with the maximum padding which does not
            result in a convolution over just padded elements.
          * snt.CAUSAL pre-pads to ensure that each output value only depends on
            input values at the same or preceding indices ("no dependence on the
            future").
          * snt.REVERSE_CAUSAL post-pads to ensure that each output value only
            depends on input values at the same or *greater* indices ("no
            dependence on the past").
          If you use the same padding for all dimensions, and it is one of SAME
          or VALID, then this is supported directly by the underlying
          convolution op. In all other cases, the input data will be padded
          using tf.pad before calling the convolution op.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b').
      partitioners: Optional dict containing partitioners to partition the
          filters (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes
          a single `Tensor` as an input and returns a scalar `Tensor` output,
          e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      data_format: A string. Specifies whether the channel dimension
          of the input and output is the last dimension (default, NHWC), or the
          second dimension (NCHW).
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      ValueError: If the given data_format is not a supported format (see
          `SUPPORTED_2D_DATA_FORMATS`).
      base.IncompatibleShapeError: If the given kernel shape is not an integer;
          or if the given kernel shape is not a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is not an integer; or if
          the given stride is not a sequence of two integers.
      ValueError: If the given padding is not `snt.VALID`, `snt.SAME`,
          `snt.FULL`, `snt.CAUSAL`, `snt.REVERSE_CAUSAL` or a sequence of these.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
    """
    if data_format not in SUPPORTED_2D_DATA_FORMATS:
      raise ValueError("Invalid data_format {:s}. Allowed formats "
                       "{}".format(data_format, SUPPORTED_2D_DATA_FORMATS))
    super(InPlaneConv2D, self).__init__(
        output_channels=lambda: self.input_channels,
        kernel_shape=kernel_shape,
        stride=stride, padding=padding, use_bias=use_bias,
        initializers=initializers, partitioners=partitioners,
        regularizers=regularizers, data_format=data_format,
        custom_getter=custom_getter, name=name)

  def _construct_w(self, inputs):
    """Construct the convolution weight matrix.

    Figures out the shape of the weight matrix, initialize it, and return it.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.

    Returns:
      w: A weight matrix of the same type as `inputs` and of shape
        [kernel_shape, 1, 1].
    """
    weight_shape = self._kernel_shape + (1, 1)

    if "w" not in self._initializers:
      self._initializers["w"] = create_weight_initializer(weight_shape[:2],
                                                          dtype=inputs.dtype)

    w = tf.get_variable("w",
                        shape=weight_shape,
                        dtype=inputs.dtype,
                        initializer=self._initializers["w"],
                        partitioner=self._partitioners.get("w", None),
                        regularizer=self._regularizers.get("w", None))
    return w

  def _apply_conv(self, inputs, w):
    """Apply a depthwise_conv2d operation on `inputs` using variable `w`.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.
      w: A weight matrix of the same type as `inputs`.

    Returns:
      outputs: The result of the convolution operation on `inputs`.
    """
    tiled_weights = tf.tile(w, [1, 1, self._input_channels, 1])
    outputs = tf.nn.depthwise_conv2d(inputs,
                                     tiled_weights,
                                     strides=self.stride,
                                     padding=self._conv_op_padding,
                                     data_format=self._data_format)
    return outputs


class DepthwiseConv2D(_ConvND):
  """Spatial depthwise 2D convolution module, including bias.

  This acts as a light wrapper around the TensorFlow ops
  `tf.nn.depthwise_conv2d`, abstracting away variable creation and sharing.
  """

  def __init__(self,
               channel_multiplier,
               kernel_shape,
               stride=1,
               padding=SAME,
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               data_format=DATA_FORMAT_NHWC,
               custom_getter=None,
               name="conv_2d_depthwise"):
    """Constructs a DepthwiseConv2D module.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/api_guides/python/nn#Convolution

    Args:
      channel_multiplier: Number of channels to expand convolution to. Must be
          an integer. Must be > 0. When `channel_multiplier` is set to 1, apply
          a different filter to each input channel producing one output channel
          per input channel. Numbers larger than 1 cause multiple different
          filters to be applied to each input channel, with their outputs being
          concatenated together, producing `channel_multiplier` *
          `input_channels` output channels.
      kernel_shape: Iterable with 2 elements in the following layout:
          [filter_height, filter_width] or integer that is
          used to define the list in all dimensions.
      stride: Iterable with 2 or 4 elements of kernel strides, or integer that
          is used to define stride in all dimensions. Layout of list:
          In case of 4 elements: `[1, stride_height, stride_widith, 1]`
          In case of 2 elements: `[stride_height, stride_width]`.
      padding: Padding algorithm. Either `snt.SAME`, `snt.VALID`, `snt.FULL`,
          `snt.CAUSAL`, `snt.REVERSE_CAUSAL`, or a sequence of these paddings
          of length 2.
          * snt.SAME and snt.VALID are explained in the Tensorflow docs at
            https://www.tensorflow.org/api_guides/python/nn#Convolution.
          * snt.FULL pre- and post-pads with the maximum padding which does not
            result in a convolution over just padded elements.
          * snt.CAUSAL pre-pads to ensure that each output value only depends on
            input values at the same or preceding indices ("no dependence on the
            future").
          * snt.REVERSE_CAUSAL post-pads to ensure that each output value only
            depends on input values at the same or *greater* indices ("no
            dependence on the past").
          If you use the same padding for all dimensions, and it is one of SAME
          or VALID, then this is supported directly by the underlying
          convolution op. In all other cases, the input data will be padded
          using tf.pad before calling the convolution op.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b').
      partitioners: Optional dict containing partitioners for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with key 'w') and the biases (with key 'b'). As a default, no
          regularizers are used. A regularizer should be a function that takes
          a single `Tensor` as an input and returns a scalar `Tensor` output,
          e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      data_format: A string. Specifies whether the channel dimension
          of the input and output is the last dimension (default, NHWC), or the
          second dimension ("NCHW").
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      ValueError: If `channel_multiplier` isn't of type (`numbers.Integral` or
          `tf.Dimension`).
      ValueError: If `channel_multiplier` is less than 1.
      ValueError: If the given data_format is not a supported format (see
          `SUPPORTED_2D_DATA_FORMATS`).
      base.IncompatibleShapeError: If the given kernel shape is not an integer;
          or if the given kernel shape is not a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is not an integer; or if
          the given stride is not a sequence of two integers.
      ValueError: If the given padding is not `snt.VALID`, `snt.SAME`,
          `snt.FULL`, `snt.CAUSAL`, `snt.REVERSE_CAUSAL` or a sequence of these.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
    """
    if (not isinstance(channel_multiplier, numbers.Integral) and
        not isinstance(channel_multiplier, tf.Dimension)):
      raise ValueError(("channel_multiplier ({}), must be of type "
                        "(`tf.Dimension`, `numbers.Integral`).").format(
                            channel_multiplier))
    if channel_multiplier < 1:
      raise ValueError("channel_multiplier ({}), must be >= 1".format(
          channel_multiplier))

    self._channel_multiplier = channel_multiplier

    if data_format not in SUPPORTED_2D_DATA_FORMATS:
      raise ValueError("Invalid data_format {:s}. Allowed formats "
                       "{}".format(data_format, SUPPORTED_2D_DATA_FORMATS))

    super(DepthwiseConv2D, self).__init__(
        output_channels=lambda: self._input_channels * self._channel_multiplier,
        kernel_shape=kernel_shape,
        stride=stride, padding=padding, use_bias=use_bias,
        initializers=initializers, partitioners=partitioners,
        regularizers=regularizers, data_format=data_format,
        custom_getter=custom_getter, name=name)

  def _construct_w(self, inputs):
    """Construct the convolution weight matrix.

    Figures out the shape of the weight matrix, initializes it, and returns it.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.

    Returns:
      w: A weight matrix of the same type as `inputs` and of shape
        [kernel_sizes, input_channels, channel_multiplier].
    """
    # For depthwise conv, output_channels = in_channels * channel_multiplier.
    # By default, depthwise conv applies a different filter to every input
    # channel. If channel_multiplier > 1, one input channel is used to produce
    # `channel_multiplier` outputs, which are then concatenated together.
    # This results in:
    weight_shape = self._kernel_shape + (self._input_channels,
                                         self._channel_multiplier)

    if "w" not in self._initializers:
      self._initializers["w"] = create_weight_initializer(weight_shape[:2],
                                                          dtype=inputs.dtype)

    w = tf.get_variable("w",
                        shape=weight_shape,
                        dtype=inputs.dtype,
                        initializer=self._initializers["w"],
                        partitioner=self._partitioners.get("w", None),
                        regularizer=self._regularizers.get("w", None))
    return w

  def _apply_conv(self, inputs, w):
    """Apply a depthwise_conv2d operation on `inputs` using variable `w`.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.
      w: A weight matrix of the same type as `inputs`.

    Returns:
      outputs: The result of the convolution operation on `inputs`.
    """
    outputs = tf.nn.depthwise_conv2d(inputs,
                                     w,
                                     strides=self.stride,
                                     padding=self._conv_op_padding,
                                     data_format=self._data_format)
    return outputs

  @property
  def channel_multiplier(self):
    """Returns the channel multiplier argument."""
    return self._channel_multiplier


class SeparableConv2D(_ConvND):
  """Performs an in-plane convolution to each channel independently.

  This acts as a light wrapper around the TensorFlow op
  `tf.nn.separable_conv2d`, abstracting away variable creation and sharing.
  """

  def __init__(self,
               output_channels,
               channel_multiplier,
               kernel_shape,
               stride=1,
               rate=1,
               padding=SAME,
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               data_format=DATA_FORMAT_NHWC,
               custom_getter=None,
               name="separable_conv2d"):
    """Constructs a SeparableConv2D module.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/api_guides/python/nn#Convolution

    Args:
      output_channels: Number of output channels. Must be an integer.
      channel_multiplier: Number of channels to expand pointwise (depthwise)
          convolution to. Must be an integer. Must be > 0.
          When `channel_multiplier` is set to 1, applies a different filter to
          each input channel. Numbers larger than 1 cause the filter to be
          applied to `channel_multiplier` input channels. Outputs are
          concatenated together.
      kernel_shape: List with 2 elements in the following layout:
          [filter_height, filter_width] or integer that is
          used to define the list in all dimensions.
      stride: List with 4 elements of kernel strides, or integer that is used to
          define stride in all dimensions. Layout of list:
          [1, stride_y, stride_x, 1].
      rate: Sequence of dilation rates (of size 2), or integer that is used to
          define dilation rate in all dimensions. 1 corresponds to standard 2D
          convolution, `rate > 1` corresponds to dilated convolution. Cannot be
          > 1 if any of `stride` is also > 1.
      padding: Padding algorithm. Either `snt.SAME`, `snt.VALID`, `snt.FULL`,
          `snt.CAUSAL`, `snt.REVERSE_CAUSAL`, or a sequence of these paddings
          of length 2.
          * snt.SAME and snt.VALID are explained in the Tensorflow docs at
            https://www.tensorflow.org/api_guides/python/nn#Convolution.
          * snt.FULL pre- and post-pads with the maximum padding which does not
            result in a convolution over just padded elements.
          * snt.CAUSAL pre-pads to ensure that each output value only depends on
            input values at the same or preceding indices ("no dependence on the
            future").
          * snt.REVERSE_CAUSAL post-pads to ensure that each output value only
            depends on input values at the same or *greater* indices ("no
            dependence on the past").
          If you use the same padding for all dimensions, and it is one of SAME
          or VALID, then this is supported directly by the underlying
          convolution op. In all other cases, the input data will be padded
          using tf.pad before calling the convolution op.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          keys 'w_dw' for depthwise and 'w_pw' for pointwise) or biases
          (with key 'b').
      partitioners: Optional dict containing partitioners to partition the
          filters (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with keys 'w_dw' for depthwise and 'w_pw' for pointwise) and the
          biases (with key 'b'). As a default, no regularizers are used.
          A regularizer should be a function that takes a single `Tensor` as an
          input and returns a scalar `Tensor` output, e.g. the L1 and L2
          regularizers in `tf.contrib.layers`.
      data_format: A string. Specifies whether the channel dimension
          of the input and output is the last dimension (default, NHWC), or the
          second dimension ("NCHW").
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      ValueError: If `channel_multiplier` isn't of type (`numbers.Integral` or
          `tf.Dimension`).
      ValueError: If `channel_multiplier` is less than 1.
      ValueError: If the given data_format is not a supported format (see
          `SUPPORTED_2D_DATA_FORMATS`).
      base.IncompatibleShapeError: If the given kernel shape is not an integer;
          or if the given kernel shape is not a sequence of two integers.
      base.IncompatibleShapeError: If the given stride is not an integer; or if
          the given stride is not a sequence of two integers.
      base.IncompatibleShapeError: If the given rate is not an integer; or if
          the given rate is not a sequence of two integers.
      base.IncompatibleShapeError: If a mask is a TensorFlow Tensor with
          a not fully defined shape.
      base.NotSupportedError: If rate in any dimension and the stride in any
          dimension are simultaneously > 1.
      ValueError: If the given padding is not `snt.VALID`, `snt.SAME`,
          `snt.FULL`, `snt.CAUSAL`, `snt.REVERSE_CAUSAL` or a sequence of these.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w_dw', 'w_pw' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      TypeError: If mask is given and it is not convertible to a Tensor.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
    """
    if (not isinstance(channel_multiplier, numbers.Integral) and
        not isinstance(channel_multiplier, tf.Dimension)):
      raise ValueError(("channel_multiplier ({}), must be of type "
                        "(`tf.Dimension`, `numbers.Integral`).").format(
                            channel_multiplier))
    if channel_multiplier < 1:
      raise ValueError("channel_multiplier ({}), must be >= 1".format(
          channel_multiplier))

    self._channel_multiplier = channel_multiplier

    if data_format not in SUPPORTED_2D_DATA_FORMATS:
      raise ValueError("Invalid data_format {:s}. Allowed formats "
                       "{}".format(data_format, SUPPORTED_2D_DATA_FORMATS))

    super(SeparableConv2D, self).__init__(
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride, padding=padding, rate=rate,
        use_bias=use_bias,
        initializers=initializers, partitioners=partitioners,
        regularizers=regularizers, data_format=data_format,
        custom_getter=custom_getter, name=name)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w_dw", "w_pw", "b"} if use_bias else {"w_dw", "w_pw"}

  def _construct_w(self, inputs):
    """Connects the module into the graph, with input Tensor `inputs`.

    Args:
      inputs: A 4D Tensor of shape:
          [batch_size, input_height, input_width, input_channels]
          and of type `tf.float16`, `tf.bfloat16`, `tf.float32` or `tf.float64`.

    Returns:
      A tuple of two 4D Tensors, each with the same dtype as `inputs`:
        1. w_dw, the depthwise weight matrix, of shape:
          [kernel_size, input_channels, channel_multiplier]
        2. w_pw, the pointwise weight matrix, of shape:
          [1, 1, channel_multiplier * input_channels, output_channels].
    """
    depthwise_weight_shape = self._kernel_shape + (self._input_channels,
                                                   self._channel_multiplier)
    pointwise_input_size = self._channel_multiplier * self._input_channels
    pointwise_weight_shape = (1, 1, pointwise_input_size, self._output_channels)

    if "w_dw" not in self._initializers:
      fan_in_shape = depthwise_weight_shape[:2]
      self._initializers["w_dw"] = create_weight_initializer(fan_in_shape,
                                                             dtype=inputs.dtype)

    if "w_pw" not in self._initializers:
      fan_in_shape = pointwise_weight_shape[:3]
      self._initializers["w_pw"] = create_weight_initializer(fan_in_shape,
                                                             dtype=inputs.dtype)

    w_dw = tf.get_variable(
        "w_dw",
        shape=depthwise_weight_shape,
        dtype=inputs.dtype,
        initializer=self._initializers["w_dw"],
        partitioner=self._partitioners.get("w_dw", None),
        regularizer=self._regularizers.get("w_dw", None))

    w_pw = tf.get_variable(
        "w_pw",
        shape=pointwise_weight_shape,
        dtype=inputs.dtype,
        initializer=self._initializers["w_pw"],
        partitioner=self._partitioners.get("w_pw", None),
        regularizer=self._regularizers.get("w_pw", None))

    return w_dw, w_pw

  def _apply_conv(self, inputs, w):
    """Apply a `separable_conv2d` operation on `inputs` using `w`.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.
      w: A tuple of weight matrices of the same type as `inputs`, the first
        being the depthwise weight matrix, and the second being the pointwise
        weight matrix.

    Returns:
      outputs: The result of the convolution operation on `inputs`.
    """
    w_dw, w_pw = w
    outputs = tf.nn.separable_conv2d(inputs,
                                     w_dw,
                                     w_pw,
                                     rate=self._rate,
                                     strides=self.stride,
                                     padding=self._conv_op_padding,
                                     data_format=self._data_format)
    return outputs

  @property
  def channel_multiplier(self):
    """Returns the channel multiplier argument."""
    return self._channel_multiplier

  @property
  def w_dw(self):
    """Returns the Variable containing the depthwise weight matrix."""
    self._ensure_is_connected()
    return self._w[0]

  @property
  def w_pw(self):
    """Returns the Variable containing the pointwise weight matrix."""
    self._ensure_is_connected()
    return self._w[1]


class SeparableConv1D(_ConvND):
  """Performs an in-plane convolution to each channel independently.

  This acts as a light wrapper around the TensorFlow op
  `tf.nn.separable_conv2d`, abstracting away variable creation and sharing.
  """

  def __init__(self,
               output_channels,
               channel_multiplier,
               kernel_shape,
               stride=1,
               rate=1,
               padding=SAME,
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               data_format=DATA_FORMAT_NWC,
               custom_getter=None,
               name="separable_conv1d"):
    """Constructs a SeparableConv1D module.

    See the following documentation for an explanation of VALID versus SAME
    padding modes:
    https://www.tensorflow.org/api_guides/python/nn#Convolution

    Args:
      output_channels: Number of output channels. Must be an integer.
      channel_multiplier: Number of channels to expand pointwise (depthwise)
          convolution to. Must be an integer. Must be > 0.
          When `channel_multiplier` is set to 1, applies a different filter to
          each input channel. Numbers larger than 1 cause the filter to be
          applied to `channel_multiplier` input channels. Outputs are
          concatenated together.
      kernel_shape: List with 2 elements in the following layout:
          [filter_height, filter_width] or integer that is
          used to define the list in all dimensions.
      stride: List with 4 elements of kernel strides, or integer that is used to
          define stride in all dimensions. Layout of list:
          [1, stride_y, stride_x, 1].
      rate: Sequence of dilation rates (of size 1), or integer that is used to
          define dilation rate in all dimensions. 1 corresponds to standard 1D
          convolution, `rate > 1` corresponds to dilated convolution. Cannot be
          > 1 if any of `stride` is also > 1.
      padding: Padding algorithm. Either `snt.SAME`, `snt.VALID`, `snt.FULL`,
          `snt.CAUSAL`, `snt.REVERSE_CAUSAL`, or a sequence of these paddings
          of length 1.
          * snt.SAME and snt.VALID are explained in the Tensorflow docs at
            https://www.tensorflow.org/api_guides/python/nn#Convolution.
          * snt.FULL pre- and post-pads with the maximum padding which does not
            result in a convolution over just padded elements.
          * snt.CAUSAL pre-pads to ensure that each output value only depends on
            input values at the same or preceding indices ("no dependence on the
            future").
          * snt.REVERSE_CAUSAL post-pads to ensure that each output value only
            depends on input values at the same or *greater* indices ("no
            dependence on the past").
          If you use the same padding for all dimensions, and it is one of SAME
          or VALID, then this is supported directly by the underlying
          convolution op. In all other cases, the input data will be padded
          using tf.pad before calling the convolution op.
      use_bias: Whether to include bias parameters. Default `True`.
      initializers: Optional dict containing ops to initialize the filters (with
          keys 'w_dw' for depthwise and 'w_pw' for pointwise) or biases
          (with key 'b').
      partitioners: Optional dict containing partitioners to partition the
          filters (with key 'w') or biases (with key 'b'). As a default, no
          partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
          (with keys 'w_dw' for depthwise and 'w_pw' for pointwise) and the
          biases (with key 'b'). As a default, no regularizers are used.
          A regularizer should be a function that takes a single `Tensor` as an
          input and returns a scalar `Tensor` output, e.g. the L1 and L2
          regularizers in `tf.contrib.layers`.
      data_format: A string. Specifies whether the channel dimension
          of the input and output is the last dimension (default, NWC), or the
          second dimension ("NCW").
      custom_getter: Callable or dictionary of callables to use as
          custom getters inside the module. If a dictionary, the keys
          correspond to regexes to match variable names. See the
          `tf.get_variable` documentation for information about the
          custom_getter API.
      name: Name of the module.

    Raises:
      ValueError: If `channel_multiplier` isn't of type (`numbers.Integral` or
          `tf.Dimension`).
      ValueError: If `channel_multiplier` is less than 1.
      ValueError: If the given data_format is not a supported format (see
          `SUPPORTED_1D_DATA_FORMATS`).
      base.IncompatibleShapeError: If the given kernel shape is not an integer;
          or if the given kernel shape is not a sequence of one integer.
      base.IncompatibleShapeError: If the given stride is not an integer; or if
          the given stride is not a sequence of two integers.
      base.IncompatibleShapeError: If the given rate is not an integer; or if
          the given rate is not a sequence of two integers.
      base.IncompatibleShapeError: If a mask is a TensorFlow Tensor with
          a not fully defined shape.
      base.NotSupportedError: If rate in any dimension and the stride in any
          dimension are simultaneously > 1.
      ValueError: If the given padding is not `snt.VALID`, `snt.SAME`,
          `snt.FULL`, `snt.CAUSAL`, `snt.REVERSE_CAUSAL` or a sequence of these.
      KeyError: If `initializers`, `partitioners` or `regularizers` contain any
          keys other than 'w_dw', 'w_pw' or 'b'.
      TypeError: If any of the given initializers, partitioners or regularizers
          are not callable.
      TypeError: If mask is given and it is not convertible to a Tensor.
      ValueError: If the passed-in data_format doesn't have a channel dimension.
    """
    if (not isinstance(channel_multiplier, numbers.Integral) and
        not isinstance(channel_multiplier, tf.Dimension)):
      raise ValueError(("channel_multiplier ({}), must be of type "
                        "(`tf.Dimension`, `numbers.Integral`).").format(
                            channel_multiplier))
    if channel_multiplier < 1:
      raise ValueError("channel_multiplier ({}), must be >= 1".format(
          channel_multiplier))

    self._channel_multiplier = channel_multiplier

    if data_format not in SUPPORTED_1D_DATA_FORMATS:
      raise ValueError("Invalid data_format {:s}. Allowed formats "
                       "{}".format(data_format, SUPPORTED_1D_DATA_FORMATS))

    super(SeparableConv1D, self).__init__(
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride, rate=rate, padding=padding, use_bias=use_bias,
        initializers=initializers, partitioners=partitioners,
        regularizers=regularizers, data_format=data_format,
        custom_getter=custom_getter, name=name)

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return {"w_dw", "w_pw", "b"} if use_bias else {"w_dw", "w_pw"}

  def _construct_w(self, inputs):
    """Connects the module into the graph, with input Tensor `inputs`.

    Args:
      inputs: A 4D Tensor of shape:
          [batch_size, input_height, input_width, input_channels]
          and of type `tf.float16`, `tf.bfloat16`, `tf.float32` or `tf.float64`.

    Returns:
      A tuple of two 4D Tensors, each with the same dtype as `inputs`:
        1. w_dw, the depthwise weight matrix, of shape:
          [kernel_size, input_channels, channel_multiplier]
        2. w_pw, the pointwise weight matrix, of shape:
          [1, 1, channel_multiplier * input_channels, output_channels].
    """
    depthwise_weight_shape = ((1,) + self._kernel_shape +
                              (self._input_channels, self._channel_multiplier))
    pointwise_input_size = self._channel_multiplier * self._input_channels
    pointwise_weight_shape = (1, 1, pointwise_input_size, self._output_channels)

    if "w_dw" not in self._initializers:
      fan_in_shape = depthwise_weight_shape[:2]
      self._initializers["w_dw"] = create_weight_initializer(fan_in_shape,
                                                             dtype=inputs.dtype)

    if "w_pw" not in self._initializers:
      fan_in_shape = pointwise_weight_shape[:3]
      self._initializers["w_pw"] = create_weight_initializer(fan_in_shape,
                                                             dtype=inputs.dtype)

    w_dw = tf.get_variable(
        "w_dw",
        shape=depthwise_weight_shape,
        dtype=inputs.dtype,
        initializer=self._initializers["w_dw"],
        partitioner=self._partitioners.get("w_dw", None),
        regularizer=self._regularizers.get("w_dw", None))

    w_pw = tf.get_variable(
        "w_pw",
        shape=pointwise_weight_shape,
        dtype=inputs.dtype,
        initializer=self._initializers["w_pw"],
        partitioner=self._partitioners.get("w_pw", None),
        regularizer=self._regularizers.get("w_pw", None))

    return w_dw, w_pw

  def _apply_conv(self, inputs, w):
    """Apply a `separable_conv2d` operation on `inputs` using `w`.

    Args:
      inputs: A Tensor of shape `data_format` and of type `tf.float16`,
          `tf.bfloat16`, `tf.float32` or `tf.float64`.
      w: A tuple of weight matrices of the same type as `inputs`, the first
        being the depthwise weight matrix, and the second being the pointwise
        weight matrix.

    Returns:
      outputs: The result of the convolution operation on `inputs`.
    """
    if self._data_format == DATA_FORMAT_NWC:
      h_dim = 1
      two_dim_conv_data_format = DATA_FORMAT_NHWC
    else:
      h_dim = 2
      two_dim_conv_data_format = DATA_FORMAT_NCHW

    inputs = tf.expand_dims(inputs, axis=h_dim)
    two_dim_conv_stride = self.stride[:h_dim] + (1,) + self.stride[h_dim:]

    # Height always precedes width.
    two_dim_conv_rate = (1,) + self._rate

    w_dw, w_pw = w
    outputs = tf.nn.separable_conv2d(inputs,
                                     w_dw,
                                     w_pw,
                                     strides=two_dim_conv_stride,
                                     rate=two_dim_conv_rate,
                                     padding=self._conv_op_padding,
                                     data_format=two_dim_conv_data_format)
    outputs = tf.squeeze(outputs, [h_dim])
    return outputs

  @property
  def channel_multiplier(self):
    """Returns the channel multiplier argument."""
    return self._channel_multiplier

  @property
  def w_dw(self):
    """Returns the Variable containing the depthwise weight matrix."""
    self._ensure_is_connected()
    return self._w[0]

  @property
  def w_pw(self):
    """Returns the Variable containing the pointwise weight matrix."""
    self._ensure_is_connected()
    return self._w[1]
