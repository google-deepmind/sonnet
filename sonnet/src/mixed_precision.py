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
"""Mixed Precision Decorator for Sonnet 2."""

import uuid
import contextlib

from sonnet.src import custom_getter
from sonnet.src import utils
import tensorflow as tf
import tree

# TODO(loreno): Make this a thread local variable
_mixed_precision_mode = None
_MP_SEEN_PROPERTY = '_mp_seen'


def enable(dtype):
  """Set the mixed precision mode.

  Args:
    dtype: type to cast to.
  """
  global _mixed_precision_mode
  _mixed_precision_mode = dtype


def disable():
  """Disable mixed precision training."""
  enable(None)


def _get_mixed_precision_mode():
  return _mixed_precision_mode


# TODO(loreno): Consider casting non-tensor/variable inputs
def _maybe_cast_element(x, dtype):
  if isinstance(x, (tf.Tensor, tf.Variable)) and x.dtype.is_floating:
    x = tf.cast(x, dtype)
  return x


def _maybe_cast_structure(x, dtype: tf.DType):
  return tree.map_structure(lambda x: _maybe_cast_element(x, dtype), x)


def _cast_call(f, new_dtype, args, kwargs):
  """Runs the function with all tensor/variable arguments casted."""
  # TODO(loreno): Implement more granular casting, not all Tensors/Variables
  args = _maybe_cast_structure(args, new_dtype)
  kwargs = _maybe_cast_structure(kwargs, new_dtype)

  # TODO(loreno): Remove float32 hardcode and replace with original dtype
  with custom_getter.custom_variable_getter(
      lambda x: _maybe_cast_structure(x, new_dtype)):
    ret = f(*args, **kwargs)
  return _maybe_cast_structure(ret, tf.float32)


def modes(valid_types):
  """Decorate a function to cast inputs/outputs to different precision.

  >>> support_modes = snt.mixed_precision.modes([tf.float32, tf.float16])
  >>> snt.Linear.__call__ = support_modes(snt.Linear.__call__)
  >>> mod = snt.Linear(10)
  >>> snt.mixed_precision.enable(tf.float16)
  >>> y = mod(tf.ones([1, 1]))  # First call will be done in F32.
  >>> y = mod(tf.ones([1, 1]))  # MatMul/Add will be done in F16.
  >>> snt.mixed_precision.disable()

  Args:
    valid_types: Collection of types that the function being decorated is legal
    to run in.

  Returns:
    A decorator that will cast the inputs and outputs of the decorated function
    according to the global mixed precision policy and the functions eligibility
    for mixed precision.
  """
  mp_id = uuid.uuid4()

  @utils.decorator
  def _wrapper(f, instance, args, kwargs):
    """Decorator to cast inputs and outputs for mixed precision.

    Args:
      f: function to handle mixed precision casting for.
      instance: instance of f.
      args: positional arguments to f.
      kwargs: keyword arguments to f.

    Returns:
      A wrapped version of `f` that casts input Variables and Tensors to the
      global mixed_precision_mode dtype if that dtype is legal for this function
      as determined by `valid_types`.
    """
    new_dtype = _get_mixed_precision_mode()
    if new_dtype is None or new_dtype not in valid_types:
      # TODO(loreno): consider throwing an error or doing nothing if input dtype
      # doesn't match any valid types
      return f(*args, **kwargs)

    if instance is None:
      if not _wrapper.seen_none:
        # TODO(loreno): Make this thread safe
        res = f(*args, **kwargs)
        _wrapper.seen_none = True
        return res
      return _cast_call(f, new_dtype, args, kwargs)

    else:
      seen = getattr(instance, _MP_SEEN_PROPERTY, None)
      if seen is None:
        seen = set()
        # TODO(loreno): use a weakrefset to address instances that define slots
        setattr(instance, _MP_SEEN_PROPERTY, seen)
      if mp_id not in seen:
        res = f(*args, **kwargs)
        seen.add(mp_id)
        return res
      return _cast_call(f, new_dtype, args, kwargs)

  _wrapper.seen_none = False
  return _wrapper


@contextlib.contextmanager
def scope(dtype: tf.DType):
  """Temporarily set the global mixed precision type to dtype.

  The global type is reset to its original value when the context is exited.::

      snt.mixed_precision.enable(tf.float32)
      support_modes = snt.mixed_precision.modes([tf.float32, tf.float16])
      snt.Linear.__call__ = support_modes(snt.Linear.__call__)
      mod = snt.Linear(10)

      with snt.mixed_precision.scope(tf.float16):
          y = mod(tf.ones([1, 1]))  # First call will be done in F32.
          y = mod(tf.ones([1, 1]))  # MatMul/Add will be done in F16.
      y = mod(tf.ones([1, 1]))  # Outside the scope will be done in F32.

  Args:
    dtype: type to set the mixed precision mode to.

  Yields:
    Nothing. This is required for contextlib.contextmanager.
  """
  # TODO(petebu) Make this a doctest once python2 is deprecated
  old_mode = _get_mixed_precision_mode()
  enable(dtype)
  try:
    yield
  finally:
    enable(old_mode)
