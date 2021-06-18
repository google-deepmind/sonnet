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
"""DNC util ops and modules."""

import numpy as np
import tensorflow as tf
import tree


def segment_dim(inputs, dim, shapes):
  """Returns tuple of Tensors output from segmenting input Tensor along dim.

  The returned tuple of Tensors produced by 'segmenting' the Tensor along a
  certain dimension can be transformed to specified shapes.

  Example:
      input_tensor = tf.placeholder([2, 14, 3])
      one, two = segment_dim(input_tensor, dim=1,
                             shapes=[TensorShape([3, 3]), TensorShape([5])])
      # one is a [2, 3, 3, 3] Tensor and two is a [2, 5, 3] Tensor.

  Args:
    inputs: `Tensor` to segment.
    dim: dimension of the Tensor to operate on. Negative numbers count back from
      the end of the dimensions.
    shapes: list of TensorShapes of the output 'segments' to produce.

  Returns:
    Tuple with resulting Tensors.

  Raises:
    ValueError: if the dim used at initialization is invalid. The valid range is
    (-d, d], where d is the number of dimensions of the input tensor.
  """
  inputs_shape = inputs.shape
  ndims = inputs_shape.ndims
  dynamic_shape = tf.shape(inputs)
  shape_as_list = [
      dynamic_shape[i] if s is None else s
      for i, s in enumerate(inputs_shape.as_list())
  ]

  if dim >= ndims or dim < -ndims:
    message = 'Invalid dims ({:d})'.format(dim)
    raise ValueError(message)

  pre_shape = shape_as_list[:dim]
  if dim == -1:
    post_shape = []
  else:
    post_shape = shape_as_list[(dim + 1):]

  slice_begin = [0] * ndims
  slice_size = [-1] * ndims

  segments = []
  for shape in shapes:
    num_elements = shape.num_elements()
    slice_size[dim] = num_elements
    flat_slice = tf.slice(inputs, slice_begin, slice_size)

    final_shape = pre_shape + shape.as_list() + post_shape
    segments.append(tf.reshape(flat_slice, final_shape))
    slice_begin[dim] += num_elements

  return tuple(segments)


def batch_invert_permutation(permutations):
  """Returns batched `tf.invert_permutation` for every row in `permutations`."""
  unpacked = tf.unstack(permutations)
  inverses = [
      tf.math.invert_permutation(permutation) for permutation in unpacked
  ]
  return tf.stack(inverses)


def batch_gather(values, indices):
  """Returns batched `tf.gather` for every row in the input."""
  unpacked = zip(tf.unstack(values), tf.unstack(indices))
  result = [tf.gather(value, index) for value, index in unpacked]
  return tf.stack(result)


def one_hot(length, index):
  """Return an nd array of given `length` filled with 0s and a 1 at `index`."""
  result = np.zeros(length)
  result[index] = 1
  return result


def apply_linear(inputs, linear_modules, activation=tf.identity):
  """Computes linear, allowing for tuple inputs (processed in parallel).

  If inputs is a tuple, the linear modules must be a tuple or list of the same
  length.

  Args:
    inputs: tensor or list / tuple of 2 tensors.
    linear_modules: sonnet module, or list / tuple of 2 sonnet modules.
    activation: function to call as activation, default is identity.

  Returns:
    output Tensor from one / both linear modules.
  """
  tree.assert_same_structure(inputs, linear_modules)
  if isinstance(inputs, (tuple, list)):
    assert len(inputs) == len(linear_modules) == 2, (
        'if inputs is a list, must be length 2 and match length of linears')
    return apply_split_linear(
        linear_modules[0],
        linear_modules[1],
        inputs[0],
        inputs[1],
        activation=activation)
  else:
    return activation(linear_modules(inputs))


def apply_split_linear(lin_module_1,
                       lin_module_2,
                       input1,
                       input2,
                       activation=None):
  """Returns a linear output of two inputs, run independently and summed."""
  output_1 = lin_module_1(input1)
  output_2 = lin_module_2(input2)
  summed_output = output_1 + output_2
  if activation is not None:
    summed_output = activation(summed_output)
  return summed_output
