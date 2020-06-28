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
"""Utils for Recurrent Neural Network cores."""

import tensorflow.compat.v1 as tf1


def _check_inputs_dtype(inputs, expected_dtype):
  if inputs.dtype is not expected_dtype:
    raise TypeError("inputs must have dtype {!r}, got {!r}".format(
        expected_dtype, inputs.dtype))
  return expected_dtype


def _safe_where(condition, x, y):  # pylint: disable=g-doc-args
  """`tf.where` which allows scalar inputs."""
  if x.shape.rank == 0:
    # This is to match the `tf.nn.*_rnn` behavior. In general, we might
    # want to branch on `tf.reduce_all(condition)`.
    return y
  # TODO(tomhennigan) Broadcasting with SelectV2 is currently broken.
  return tf1.where(condition, x, y)