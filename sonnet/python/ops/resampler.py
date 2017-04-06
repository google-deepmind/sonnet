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

"""Tensorflow op performing differentiable resampling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import ops

try:
  from sonnet.python.ops import gen_resampler as _gen_resampler  # pylint: disable=g-import-not-at-top
except ImportError:
  _gen_resampler = None
else:
  # Link the shared object.
  _resampler_so = tf.load_op_library(
      tf.resource_loader.get_path_to_datafile("_resampler.so"))


def resampler_is_available():
  return _gen_resampler is not None


def resampler(data, warp, name="resampler"):
  """Resamples input data at user defined coordinates.

  The resampler currently only supports bilinear interpolation of 2D data.

  Args:
    data: Tensor of shape `[batch_size, data_height, data_width,
      data_num_channels]` containing 2D data that will be resampled.
    warp: Tensor of minimum rank 2 containing the coordinates at which
      resampling will be performed. Since only bilinear interpolation is
      currently supported, the last dimension of the `warp` tensor must be 2.
    name: Optional name of the op.

  Returns:
    Tensor of resampled values from `data`. The output tensor shape is
    determined by the shape of the warp tensor. For example, if `data` is of
    shape `[batch_size, data_height, data_width, data_num_channels]` and warp of
    shape `[batch_size, dim_0, ... , dim_n, 2]` the output will be of shape
    `[batch_size, dim_0, ... , dim_n, data_num_channels]`.

  Raises:
    ImportError: if the wrapper generated during compilation is not present when
    the function is called.
  """
  if not resampler_is_available():
    raise ImportError("_gen_resampler could not be imported.")
  with ops.name_scope(name, "resampler", [data, warp]):
    data_tensor = ops.convert_to_tensor(data, name="data")
    warp_tensor = ops.convert_to_tensor(warp, name="warp")
    return _gen_resampler.resampler(data_tensor, warp_tensor)


@ops.RegisterGradient("Resampler")
def _resampler_grad(op, grad_output):
  data, warp = op.inputs
  grad_output_tensor = ops.convert_to_tensor(grad_output, name="grad_output")
  return _gen_resampler.resampler_grad(data, warp, grad_output_tensor)


ops.NotDifferentiable("ResamplerGrad")
