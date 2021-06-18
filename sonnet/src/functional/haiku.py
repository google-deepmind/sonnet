# Copyright 2020 The Sonnet Authors. All Rights Reserved.
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
"""Implements part of the Haiku ("Sonnet for JAX") API in TensorFlow 2."""

import collections
import functools
import itertools
import threading

import contextlib
from sonnet.src.functional import utils
import tensorflow as tf

Transformed = collections.namedtuple("Transformed", ("init", "apply"))
TransformedWithState = collections.namedtuple("TransformedWithState",
                                              ("init", "apply"))

# pylint: disable=not-context-manager


class TensorVariableCallbacks(threading.local):
  """Holds callbacks that are notified when TensorVariable are used."""

  instance = None  # Thread local singleton instance.

  def __init__(self):
    super().__init__()
    self._recording = False
    self._callbacks = []

  def notify(self, variable):
    if self._recording:
      assert isinstance(variable, TensorVariable)
      for callback in self._callbacks:
        callback(variable)

  @contextlib.contextmanager
  def __call__(self, callback):
    self._callbacks.append(callback)
    recording = self._recording
    try:
      self._recording = True
      yield
    finally:
      assert self._callbacks.pop() is callback
      self._recording = recording

TensorVariableCallbacks.instance = TensorVariableCallbacks()


def notify(f):
  """Wraps `f` such that callbacks are notified about it being called."""
  @functools.wraps(f)
  def wrapper(self, *args, **kwargs):
    TensorVariableCallbacks.instance.notify(self)
    return f(self, *args, **kwargs)  # pytype: disable=wrong-arg-count
  return wrapper


def defer_property(name):
  return property(fget=notify(lambda self: getattr(self.tensor_value, name)))


def safe_read_tensor_value(variable):
  """Reads variable value or raises an exception."""

  value = variable.tensor_value
  if value is None:
    raise ValueError("".join((
        "Attempted to read a TensorVariable in a context where it has no ",
        "value. This commonly happens for one of two reasons:",
        "",
        "   1) You created a model in one transformed function and directly",
        "      accessed the model variables (e.g. via `model.variables` or"
        "      `model.w`) inside another transformed function.",
        "   2) You are trying to read a model variable outside of a",
        "      transformed function.",
        "",
        "For (1) you can safely do this if you do not read the value of the",
        "variable (e.g. you just use metadata like `v.shape` or `v.dtype`).",
        "If you want to read the value of the variable then you must pass in",
        "the value (e.g. pass the result of `f.init(..)`).",
        "",
        "For (2) to read variable values inspect the result of a transformed",
        "function (e.g. look at the `params` dictionary returned from ",
        "`f.init(..)`).")))

  return value


def defer_read():
  return property(
      fget=notify(lambda self: (lambda: safe_read_tensor_value(self))))


def defer_raise_notimplemented():
  def _raise_notimplemented():
    raise NotImplementedError

  return property(fget=notify(_raise_notimplemented))


def defer_indexed(f):
  return property(fget=notify(lambda self, i: f(self, i.indices, i.values)))


def defer_assign(map_fn=None):
  """Returns a function implementing notify+assign."""
  @notify
  def wrapped(self, v):
    if v is not None:
      v = tf.convert_to_tensor(v, dtype=self.dtype)
    if map_fn is not None:
      v = map_fn(self.tensor_value, v)
    if self.initial_tensor_value is None:
      self.initial_tensor_value = v
    self.tensor_value = v
    return v
  return wrapped


class TensorVariable(tf.Variable):
  """Implements the tf.Variable API but backed by a tf.Tensor."""

  def __init__(self, value, trainable, name=None):
    # NOTE: Intentionally not calling super ctor.
    self.initial_tensor_value = value
    self.tensor_value = value
    self._trainable = trainable
    self._name = name
    self._shape = value.shape
    self._dtype = value.dtype
    self._device = value.device

  # Properties.
  # NOTE: These do not notify since they do not result in TensorFlow operations.
  shape = property(fget=lambda self: self._shape)
  dtype = property(fget=lambda self: self._dtype)
  trainable = property(fget=lambda self: self._trainable)
  name = property(fget=lambda self: self._name)
  device = property(fget=lambda self: self._device)

  # Dense assign.
  assign = defer_assign()
  assign_add = defer_assign(tf.add)
  assign_sub = defer_assign(tf.subtract)

  # Sparse assign.
  batch_scatter_update = defer_raise_notimplemented()
  scatter_add = defer_raise_notimplemented()
  scatter_div = defer_raise_notimplemented()
  scatter_max = defer_raise_notimplemented()
  scatter_min = defer_raise_notimplemented()
  scatter_mul = defer_raise_notimplemented()
  scatter_sub = defer_raise_notimplemented()
  scatter_update = defer_raise_notimplemented()
  scatter_nd_add = defer_indexed(tf.tensor_scatter_nd_add)
  scatter_nd_sub = defer_indexed(tf.tensor_scatter_nd_sub)
  scatter_nd_update = defer_indexed(tf.tensor_scatter_nd_update)

  # Load not supported.
  load = defer_raise_notimplemented()

  # Shape ops.
  set_shape = defer_property("set_shape")
  get_shape = defer_property("get_shape")

  # Read dense.
  initialized_value = property(
      fget=notify(lambda self: self.initial_tensor_value))
  read_value = defer_read()
  numpy = defer_property("numpy")
  value = defer_read()
  eval = defer_property("eval")

  # Read sparse.
  gather_nd = defer_indexed(tf.gather_nd)
  sparse_read = defer_indexed(tf.gather)

  # Serialize.
  to_proto = defer_raise_notimplemented()

  # Misc.
  count_up_to = defer_raise_notimplemented()

  def __repr__(self):
    return "TensorVariable(shape={}, dtype={}, name={!r})".format(
        list(self.shape), self.dtype.name, self.name)

  __str__ = __repr__

  # Math ops.
  __add__ = defer_property("__add__")
  __sub__ = defer_property("__sub__")
  __mul__ = defer_property("__mul__")
  __div__ = defer_property("__div__")


@functools.partial(tf.register_tensor_conversion_function, TensorVariable)
@notify
def tv_to_tensor(value, dtype=None, name=None, as_ref=None):
  """Converts a TensorVariable to a tf.Tensor."""
  del as_ref
  tensor_value = value.tensor_value
  if tensor_value is None:
    # TODO(tomhennigan) We should probably not notify in this case.
    tensor_value = tf.zeros(value.shape, dtype=value.dtype)
  if dtype is not None:
    tensor_value = tf.cast(tensor_value, dtype=dtype, name=name)
  return tensor_value


def create_tensor_variables():
  """Defines a scope in which `TensorVariable`s are created.

  >>> with snt.functional.variables():
  ...   v = tf.Variable(tf.ones([]), name="v")
  >>> v.tensor_value
  <tf.Tensor: ... numpy=1.0>

  Returns:
    A context manager that forces tf.Variable to create TensorVariables.
  """

  def getter(next_getter, **kwargs):
    del next_getter
    initial_value = tf.convert_to_tensor(kwargs["initial_value"])
    trainable = utils.first_non_none(kwargs["trainable"], True)
    name = utils.first_non_none(kwargs["name"], "Variable")
    name = utils.get_name_scope() + name + ":0"
    return TensorVariable(initial_value, trainable=trainable, name=name)

  return tf.variable_creator_scope(getter)

variables = create_tensor_variables


@contextlib.contextmanager
def track_tensor_variables():
  tensor_variables = []
  with TensorVariableCallbacks.instance(tensor_variables.append):  # pylint: disable=not-callable
    yield tensor_variables


@contextlib.contextmanager
def track_new_variables():
  new_variables = []
  def getter(next_getter, *args, **kwargs):
    var = next_getter(*args, **kwargs)
    new_variables.append(var)
    return var

  with tf.variable_creator_scope(getter):
    yield new_variables


@contextlib.contextmanager
def track_initial_state():
  var_state = {}
  def callback(v):
    r = v.ref()
    if r not in var_state:
      var_state[r] = (v.initial_tensor_value, v.tensor_value)

  with TensorVariableCallbacks.instance(callback):  # pylint: disable=not-callable
    yield var_state


def initial_value_by_ref(tf_variables):
  # TODO(tomhennigan) Consider rolling own ref class comparing by name/shape.
  return {v.ref(): v.initial_tensor_value for v in tf_variables}


def final_value_by_ref(tf_variables):
  # TODO(tomhennigan) Consider rolling own ref class comparing by name/shape.
  return {v.ref(): v.tensor_value for v in tf_variables}


def transform(f) -> Transformed:
  """Transforms a function using Sonnet modules into a pair of pure functions.

  The first thing to do is to create some `snt.Module` instances:

  >>> with snt.functional.variables():
  ...   a = snt.Linear(10, name="a")
  ...   b = snt.Linear(10, name="b")

  Next, define some function that creates and applies modules:

  >>> def f(x):
  ...   return a(x) + b(x)

  Now we can convert that function into a pair of functions that allow us to
  lift all the parameters out of the function (`f.ini`) and apply the function
  with a given set of parameters (`f.apply`):

  >>> f = snt.functional.transform(f)

  To get the initial state of the module call `f.init` with an example input:

  >>> x = tf.ones([1, 1])
  >>> params = f.init(x)
  >>> params
  {<...>: <tf.Tensor: ...>,
   <...>: <tf.Tensor: ...>,
   <...>: <tf.Tensor: ...>,
   <...>: <tf.Tensor: ...>}

  You can then apply the function with the given parameters by calling
  `f.apply`:

  >>> f.apply(params, x)
  <tf.Tensor: ...>

  It is expected that your program will at some point produce updated parameters
  and you will want to re-apply `f.apply`. You can do this by calling
  `f.apply` with different parameters:

  >>> new_params = tree.map_structure(lambda p: p + 1, params)
  >>> f.apply(new_params, x)
  <tf.Tensor: ...>

  If your network contains non-trainable state (e.g. moving averages) then you
  will need to use :func:`transform_with_state`.

  Args:
    f: A function closing over `Module` instances.

  Returns:
    A transformed function with `init` and `apply`. See docstring for details.
  """
  return without_state(transform_with_state(f))


def transform_with_state(f) -> TransformedWithState:
  r"""Like :func:`transform` but supporting non-trainable state.

  See :func:`transform` for more details.

  It is possible for the network to maintain internal state (e.g. for a module
  like `BatchNorm` that may want to maintain a moving average):

  >>> with snt.functional.variables():
  ...   ema = snt.ExponentialMovingAverage(decay=0.5)

  >>> f = snt.functional.transform_with_state(ema)

  When initializing this network we are returned the parameters (any "trainable"
  :tf:`Variable`\ s) and all other state (any non-trainable :tf:`Variable`\ s):

  >>> params, state = f.init(3.0)
  >>> params
  {}
  >>> state
  {<...>: <tf.Tensor: ... numpy=0>,
   <...>: <tf.Tensor: ... numpy=0.0>,
   <...>: <tf.Tensor: ... numpy=0.0>}

  To apply the network we simply call it and get back updated values for our
  non-trainable state:

  >>> y, state = f.apply(params, state, 3.0)
  >>> y.numpy()
  3.0

  >>> y, state = f.apply(params, state, 6.0)
  >>> y.numpy()
  5.0

  Args:
    f: A function closing over `Module` instances.

  Returns:
    A transformed function with `init` and `apply`. See docstring for details.
  """
  def init_fn(*args, **kwargs):
    """Applies `f(*a, **k)` and extracts initial variable values."""
    with create_tensor_variables(), \
         track_new_variables() as new_variables, \
         track_initial_state() as prev_var_state, \
         track_tensor_variables() as tensor_variables:

      # NOTE: Intentionally discarding result.
      f(*args, **kwargs)

    params = initial_value_by_ref(v for v in tensor_variables if v.trainable)
    state = initial_value_by_ref(v for v in tensor_variables if not v.trainable)

    # Reset variable values.
    new_variables = {v.ref() for v in new_variables}
    for v in tensor_variables:
      r = v.ref()
      if r in new_variables:
        # Variables created inside the function have their values nullified.
        initial_tensor_value, tensor_value = None, None
      else:
        # Variables that already existed have their value reset.
        initial_tensor_value, tensor_value = prev_var_state[r]
      v.initial_tensor_value = initial_tensor_value
      v.tensor_value = tensor_value

    return params, state

  def apply_fn(params, state, *args, **kwargs):
    """Applies `f(*a, **k)` with variable values passed in."""
    initial_values = {}
    for r, t in itertools.chain(params.items(), state.items()):
      v = r.deref()
      initial_values[r] = (v.tensor_value, v.initial_tensor_value)
      v.assign(t)

    try:
      with track_new_variables() as new_variables:
        out = f(*args, **kwargs)
      if new_variables:
        raise ValueError("Apply function cannot create new variables.")
      state = final_value_by_ref(p.deref() for p in state.keys())
      return out, state

    finally:
      # Reset values to their initial state.
      for r, (tensor_value, initial_tensor_value) in initial_values.items():
        v = r.deref()
        v.tensor_value = tensor_value
        v.initial_tensor_value = initial_tensor_value

  return TransformedWithState(init=init_fn, apply=apply_fn)


def without_state(with_state: TransformedWithState) -> Transformed:
  """Returns init/apply functions that ignore state."""

  def init_fn(*args, **kwargs):
    params, state = with_state.init(*args, **kwargs)
    if state:
      raise ValueError("Stateful networks must use `transform_with_state(f)`")
    return params

  def apply_fn(params, *args, **kwargs):
    y, state = with_state.apply(params, {}, *args, **kwargs)
    if state:
      raise ValueError("Stateful networks must use `transform_with_state(f)`")
    return y

  return Transformed(init_fn, apply_fn)
