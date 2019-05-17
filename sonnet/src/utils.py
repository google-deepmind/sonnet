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

"""Utils for Sonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import inspect
import re

import tensorflow as tf


def replicate(element, num_times, name):
  """Replicates entry in `element` `num_times` if needed."""
  if not isinstance(element, collections.Sequence):
    return (element,) * num_times
  elif len(element) == 1:
    return tuple(element * num_times)
  elif len(element) == num_times:
    return tuple(element)
  raise TypeError(
      "{} must be a scalar or sequence of length 1 or sequence of length {}."
      .format(name, num_times))


def _is_object(f):
  return not inspect.isfunction(f) and not inspect.ismethod(f)


# TODO(b/123870292) Remove this and use wrapt.decorator when supported by TF.
def decorator(decorator_fn):
  """Returns a wrapt style decorator."""
  @functools.wraps(decorator_fn)
  def _decorator(f):
    """Wraps f such that it returns the result of applying decorator_fn."""
    if _is_object(f):
      @functools.wraps(f.__call__)
      def _decorate_object(*args, **kwargs):
        return decorator_fn(f.__call__, f, args, kwargs)
      return _decorate_object

    if inspect.ismethod(f):
      @functools.wraps(f)
      def _decorate_bound_method(*args, **kwargs):
        return decorator_fn(f, f.__self__, args, kwargs)
      return _decorate_bound_method

    argspec = inspect.getargspec(f)
    if argspec.args and argspec.args[0] == "self":
      @functools.wraps(f)
      def _decorate_unbound_method(self, *args, **kwargs):
        bound_method = f.__get__(self, self.__class__)
        return decorator_fn(bound_method, self, args, kwargs)
      return _decorate_unbound_method

    @functools.wraps(f)
    def _decorate_fn(*args, **kwargs):
      return decorator_fn(f, None, args, kwargs)
    return _decorate_fn

  return _decorator


_SPATIAL_CHANNELS_FIRST = re.compile("^NC[^C]*$")
_SPATIAL_CHANNELS_LAST = re.compile("^N[^C]*C$")
_SEQUENTIAL = re.compile("^((BT)|(TB))[^D]*D$")


def get_channel_index(data_format):
  """Returns the channel index when given a valid data format.

  Args:
    data_format: String, the data format to get the channel index from. Valid
      data formats are spatial (e.g.`NCHW`), sequential (e.g. `BTHWD`),
      `channels_first` and `channels_last`).

  Returns:
    The channel index as an int - either 1 or -1.

  Raises:
    ValueError: If the data format is unrecognised.
  """
  if data_format == "channels_first":
    return 1
  if data_format == "channels_last":
    return -1
  if _SPATIAL_CHANNELS_FIRST.match(data_format):
    return 1
  if _SPATIAL_CHANNELS_LAST.match(data_format):
    return -1
  if _SEQUENTIAL.match(data_format):
    return -1
  raise ValueError(
      "Unable to extract channel information from '{}'. Valid data formats are "
      "spatial (e.g.`NCHW`), sequential (e.g. `BTHWD`), `channels_first` and "
      "`channels_last`).".format(data_format))


def assert_rank(inputs, rank):
  """Asserts the rank of the input is `rank`."""
  shape = tuple(inputs.shape)
  actual_rank = len(shape)
  if rank != actual_rank:
    raise ValueError("Shape %r must have rank %d" % (shape, rank))


def assert_minimum_rank(inputs, rank):
  """Asserts the rank of the input is at least `rank`."""
  shape = tuple(inputs.shape)
  actual_rank = len(shape)
  if actual_rank < rank:
    raise ValueError("Shape %r must have rank >= %d" % (shape, rank))


def smart_autograph(f):
  """Wraps `f` such that in graph mode it uses autograph but not in eager.

  Whilst wrapping `f` in autograph is (intended to be) semantics preserving,
  some things (e.g. breakpoints) are not preserved. Using `smart_autograph`
  users can write code with eager syntax, add breakpoints and debug it as you
  might expect and still be compatible with code that uses
  `@tf.function(autograph=False)`.

      >>> @smart_autograph
      ... def f(x):
      ...   if x > 0:
      ...     y = x * x
      ...   else:
      ...     y = -x
      ...   return y

      >>> f = tf.function(f, autograph=False)
      >>> f(tf.constant(2))
      <tf.Tensor: ... numpy=4>

  Args:
    f: A function to wrap conditionally in `tf.autograph`.

  Returns:
    A wrapper for `f` that dispatches to the original or autograph version of f.
  """
  f_autograph = tf.autograph.to_graph(f)

  @functools.wraps(f)
  def smart_autograph_wrapper(*args, **kwargs):
    if tf.executing_eagerly():
      return f(*args, **kwargs)
    else:
      return f_autograph(*args, **kwargs)

  return smart_autograph_wrapper
