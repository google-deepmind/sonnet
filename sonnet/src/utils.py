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

import collections.abc
import functools
import inspect
import re
import threading
from typing import Any, Callable, Dict, Generic, Optional, Sequence, Tuple, TypeVar, Union

from absl import logging
from sonnet.src import initializers
import tabulate
import tensorflow as tf

T = TypeVar("T")


def replicate(
    element: Union[T, Sequence[T]],
    num_times: int,
    name: str,
) -> Tuple[T]:
  """Replicates entry in `element` `num_times` if needed."""
  if not isinstance(element, collections.abc.Sequence):
    return (element,) * num_times
  elif len(element) == 1:
    return tuple(element * num_times)
  elif len(element) == num_times:
    return tuple(element)
  raise TypeError(
      "{} must be a scalar or sequence of length 1 or sequence of length {}."
      .format(name, num_times))


def _is_object(f: Any) -> bool:
  return not inspect.isfunction(f) and not inspect.ismethod(f)


# TODO(b/123870292) Remove this and use wrapt.decorator when supported by TF.
def decorator(
    decorator_fn: Callable[[T, Any, Sequence[Any], Dict[str, Any]], Any],) -> T:
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

    argspec = inspect.getfullargspec(f)
    if argspec.args and argspec.args[0] == "self":

      # TODO(b/123870292): Bound methods with multiple decorators can be
      # detected as unbound too, which breaks tf.function usage.
      # Attempt migrating to wrapt instead of using this custom approach.
      def _decorate_unbound_method(self, *args, **kwargs):
        bound_method = f.__get__(self, self.__class__)  # pytype: disable=attribute-error
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


def get_channel_index(data_format: str) -> int:
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


def assert_rank(inputs, rank: int):
  """Asserts the rank of the input is `rank`."""
  shape = tuple(inputs.shape)
  actual_rank = len(shape)
  if rank != actual_rank:
    raise ValueError("Shape %r must have rank %d" % (shape, rank))


def assert_minimum_rank(inputs, rank: int):
  """Asserts the rank of the input is at least `rank`."""
  shape = tuple(inputs.shape)
  actual_rank = len(shape)
  if actual_rank < rank:
    raise ValueError("Shape %r must have rank >= %d" % (shape, rank))


def _synchronized(f: Callable[..., T]) -> Callable[..., T]:
  """Returns a version of f that can only be called by one thread at a once."""
  lock = threading.RLock()
  @functools.wraps(f)
  def wrapper(*a, **k):
    with lock:
      return f(*a, **k)
  return wrapper


def smart_autograph(f: Callable[..., T]) -> Callable[..., T]:
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
  cache = []

  @_synchronized
  def f_autograph():
    if not cache:
      cache.append(tf.autograph.to_graph(f))
    return cache[0]

  @functools.wraps(f)
  def smart_autograph_wrapper(*args, **kwargs):
    if tf.executing_eagerly():
      return f(*args, **kwargs)
    else:
      return f_autograph()(*args, **kwargs)

  return smart_autograph_wrapper


def variable_like(inputs: Union[tf.Tensor, tf.Variable],
                  initializer: initializers.Initializer = initializers.Zeros(),
                  trainable: Optional[bool] = None,
                  name: Optional[str] = None) -> tf.Variable:
  """Creates a new variable with the same shape/dtype/device as the input."""
  if trainable is None:
    trainable = getattr(inputs, "trainable", None)
  if name is None:
    name = getattr(inputs, "name", "Variable").split(":")[0]
  with tf.device(inputs.device):
    initial_value = initializer(inputs.shape, inputs.dtype)
    return tf.Variable(initial_value, trainable=trainable, name=name)


def _render_spec(shape: tf.TensorShape, dtype: tf.DType) -> str:
  """Renders the given shape/dtype as a short specification."""

  format_map = {
      tf.float16: "f16",
      tf.float32: "f32",
      tf.float64: "f64",
      tf.bfloat16: "bf16",
      tf.complex64: "c64",
      tf.complex128: "c128",
      tf.uint8: "u8",
      tf.uint16: "u16",
      tf.uint32: "u32",
      tf.uint64: "u64",
      tf.int8: "i8",
      tf.int16: "i16",
      tf.int32: "i32",
      tf.int64: "i64",
      tf.qint8: "qi8",
      tf.qint16: "qi16",
      tf.qint32: "qi32",
      tf.quint8: "qu8",
      tf.quint16: "qu16",
  }

  return "{dtype}[{shape}]".format(
      dtype=format_map.get(dtype, dtype.name),
      shape=",".join(str(d) for d in shape))


def _simple_device(var: tf.Variable) -> str:
  device = tf.DeviceSpec.from_string(var.device)
  if device.job == "localhost" and device.replica == 0 and device.task == 0:
    if device.device_index == 0:
      return device.device_type
    else:
      return "{} {}".format(device.device_type, device.device_index)
  return device


def _name_scope_then_rank(var: tf.Variable):
  name_scope = "/".join(var.name.split("/")[:-1])
  rank = len(var.shape)
  return (name_scope, -rank, var.name)


def format_variables(variables: Sequence[tf.Variable],
                     tablefmt: str = "orgtbl") -> str:
  """Takes a collection of variables and formats it as a table."""
  rows = []
  for var in sorted(variables, key=_name_scope_then_rank):
    name = var.name.split(":")[0]  # Remove the ":0" suffix.
    spec = _render_spec(var.shape, var.dtype)
    trainable = str(var.trainable)
    device = _simple_device(var)
    rows.append((name, spec, trainable, device))
  return tabulate.tabulate(
      rows,
      headers=("Variable", "Spec", "Trainable", "Device"),
      tablefmt=tablefmt)


def log_variables(variables: Sequence[tf.Variable]):
  """Logs variable information.

  This function logs the name, shape, type, trainability, and device for a
  given iterable of variables.

  Args:
    variables: iterable of variables (e.g., `module.variables`, if `module` is a
      `snt.Module` instance).
  """
  for line in format_variables(variables).split("\n"):
    logging.info(line)


@functools.total_ordering
class CompareById(Generic[T]):
  """Container providing hash/eq based on object id."""

  def __init__(self, wrapped: T):
    self.wrapped = wrapped

  def __hash__(self):
    # NOTE: `dict` has special casing to allow for hash values that are
    # sequential ints (since `hash(i: int) -> i`) so the using `id` as a hash
    # code (at least with `dict` and `set`) does not have a big performance
    # penalty.
    # https://github.com/python/cpython/blob/master/Objects/dictobject.c#L135
    return id(self.wrapped)

  def __eq__(self, other):
    if other is None:
      return False
    return self.wrapped is getattr(other, "wrapped", None)

  def __lt__(self, other):
    return id(self.wrapped) < id(getattr(other, "wrapped", None))
