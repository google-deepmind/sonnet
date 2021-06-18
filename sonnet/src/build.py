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
"""Utility function to build Sonnet modules."""

from typing import Any, Callable

import tensorflow as tf
import tree


def _int_or_none(o):
  return isinstance(o, (int, type(None)))


def _promote_shapes(o):
  """Promotes lists of ints/Nones to :tf:`TensorSpec` instances."""
  if isinstance(o, (list, tuple)) and all(_int_or_none(e) for e in o):
    return tf.TensorSpec(o)
  return o


def _maybe_tensor_spec(shape, dtype):
  return tf.TensorSpec(shape, dtype) if dtype is not None else None


# TODO(tomhennigan) Use TensorNest in types here.
def build(
    f: Callable[..., Any],
    *args,
    **kwargs
):
  r"""Builds a module by creating all parameters but not computing any output.

      >>> mod = snt.nets.MLP([1000, 10])
      >>> snt.build(mod, [None, 28 * 28])
      TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
      >>> mod.variables
      (<tf.Variable 'mlp/linear_0/b:0' shape=(1000,) ...>,
       <tf.Variable 'mlp/linear_0/w:0' shape=(784, 1000) ...>,
       <tf.Variable 'mlp/linear_1/b:0' shape=(10,) ...>,
       <tf.Variable 'mlp/linear_1/w:0' shape=(1000, 10) ...>)

  Args:
    f: A function or callable :class:`Module` that will create variables.
    *args: Positional arguments to supply to ``f``. Note that positional
      arguments that are sequences of None/ints are converted to
      :tf:`TensorSpec` instances.
    **kwargs: Keyword arguments to pass to the module.

  Returns:
    The output of ``f`` with any :tf:`Tensor`\ s replaced by :tf:`TensorSpec`.
  """
  f = tf.function(f)
  args = map(_promote_shapes, args)
  # NOTE: We use a concrete function to ensure that weights are created and
  # initialized, but other stateful ops (e.g. updating weights) are not.
  cf = f.get_concrete_function(*args, **kwargs)
  return tree.map_structure(_maybe_tensor_spec, cf.output_shapes,
                            cf.output_dtypes)
