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

"""A checkpoint-restoring Tensorflow initializer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import io_ops


class _Restore(init_ops.Initializer):
  """Initializer that restores tensors from a checkpoint."""

  def __init__(self, filename, var_name, scope=None):
    """Construct a new restoring initializer.

    Will read from the checkpoint from the SSTables file `filename` using
    the RestoreV2 Tensorflow op.

    The actual variable read from the checkpoint will be
    `scope_name` + '/' + `var_name` (or just `var_name` if `scope_name` is
    empty), where `scope_name` is given by one of

    (1) The current scope's name at the point where the initializer gets called,
        if the `scope` argument to this constructor is None,
    (2) If `scope` is callable, the result of applying it to the current scope's
        name,
    (3) Otherwise, the `scope` argument to this constructor itself.

    Args:
      filename: Name of an SSTables entry where the checkpoint is hosted.
      var_name: Name of the variable to restore.
      scope: The variable scope's name of the variable to restore, see above.
    """
    self._filename = filename
    self._var_name = var_name
    self._scope = scope

  def _partition_spec(self, shape, partition_info):
    """Build magic (and sparsely documented) shapes_and_slices spec string."""
    if partition_info is None:
      return ''  # Empty string indicates a non-partitioned tensor.
    ssi = tf.Variable.SaveSliceInfo(
        full_name=self._var_name,
        full_shape=partition_info.full_shape,
        var_offset=partition_info.var_offset,
        var_shape=shape)
    return ssi.spec

  def __call__(self, shape, dtype=None, partition_info=None):
    # Creating different RestoreV2 ops when a single one could
    # output several tensors seems inefficient, but that's actually
    # what tf.Saver.restore_op (via tf.BaseSaverBuilder) does too.
    if self._scope is None:
      scope_name = tf.get_variable_scope().name
    elif callable(self._scope):
      scope_name = self._scope(tf.get_variable_scope().name)
    else:
      scope_name = self._scope
    tensor_name = self._var_name
    if scope_name:
      tensor_name = '{}/{}'.format(scope_name, tensor_name)
    tensor = io_ops.restore_v2(
        self._filename,
        [tensor_name],
        [self._partition_spec(shape, partition_info)],
        [dtype])[0]
    tensor.set_shape(shape)
    return tensor


# pylint: disable=invalid-name
restore_initializer = _Restore
# pylint: enable=invalid-name
