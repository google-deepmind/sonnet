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

"""Base class for TensorFlow Sonnet recurrent cores.

This file contains the Abstract Base Class for defining Recurrent Cores in
TensorFlow. A Recurrent Core is an object that holds the properties of other
`snt.Module`s and also satisfies the interface of any RNNCell in tensorflow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import warnings

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from sonnet.python.modules import base
from sonnet.python.modules import basic
import tensorflow as tf
import wrapt

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest


def _single_learnable_state(state, state_id=0, learnable=True):
  """Returns an initial (maybe learnable) state.

  This function does not create any variable scopes, and it should be called
  from a Sonnet module. This function also makes sure that all the rows of its
  `state` argument have the same value.

  Args:
    state: initial value of the initial state. It should be a tensor of at least
      two dimensions, of which the first dimension corresponds to the
      batch_size dimension. All rows of such tensor should have the same value.
    state_id: integer that uniquely identifies this state.
    learnable: boolean that indicates whether the state is learnable.

  Returns:
    The initial learnable state `Tensor`.
  """
  unpacked_state = tf.unstack(state)
  # Assert that all rows have the same values.
  assert_rows_equal = [tf.assert_equal(s, unpacked_state[0])
                       for s in unpacked_state]

  # We wish to have all the graph assertions in the graph's critical path,
  # so we include them even if the initial state is left unmodified (i.e. when
  # the state is not learnable).
  # Note: All these assertions will be run every time that data flows
  # through the graph. At that point, the control_dependencies context manager
  # makes sure that such assertions are run, and will raise an exception if any
  # fails.
  with tf.control_dependencies(assert_rows_equal):
    if not learnable:
      return state
    else:
      state_shape = state.get_shape()
      state_shape.assert_is_fully_defined()
      state_shape_list = state_shape.as_list()
      batch_size, trailing_shape = state_shape_list[0], state_shape_list[1:]

      initial_value = tf.reshape(unpacked_state[0], [1] + trailing_shape)
      initial_state_variable = tf.get_variable(
          "initial_state_%d" % state_id, dtype=initial_value.dtype,
          initializer=initial_value)

      trailing_size_repeat = [1] * len(trailing_shape)
      return tf.tile(initial_state_variable,
                     tf.constant([batch_size] + trailing_size_repeat))


def trainable_initial_state(batch_size, state_size, dtype, initializers=None,
                            regularizers=None, name=None):
  """Creates an initial state consisting of trainable variables.

  The trainable variables are created with the same shapes as the elements of
  `state_size` and are tiled to produce an initial state.

  Args:
    batch_size: An int, or scalar int32 Tensor representing the batch size.
    state_size: A `TensorShape` or nested tuple of `TensorShape`s to use for the
        shape of the trainable variables.
    dtype: The data type used to create the variables and thus initial state.
    initializers: An optional container of the same structure as `state_size`
        containing initializers for the variables.
    regularizers: An optional container of the same structure as `state_size`
        containing regularizers for the variables.
    name: optional string used to prefix the initial state variable names.

  Returns:
    A `Tensor` or nested tuple of `Tensor`s with the same size and structure
    as `state_size`, where each `Tensor` is a tiled trainable `Variable`.

  Raises:
    ValueError: if the user passes initializers that are not functions.
    ValueError: if the user passes regularizers that are not functions.
  """
  flat_state_size = nest.flatten(state_size)

  if not initializers:
    flat_initializer = tuple(tf.zeros_initializer() for _ in flat_state_size)
  else:
    nest.assert_same_structure(initializers, state_size)
    flat_initializer = nest.flatten(initializers)
    if not all([callable(init) for init in flat_initializer]):
      raise ValueError("Not all the passed initializers are callable objects.")

  if not regularizers:
    flat_regularizer = tuple({} for _ in flat_state_size)
  else:
    nest.assert_same_structure(regularizers, state_size)
    flat_regularizer = nest.flatten(regularizers)
    if not all([callable(regularizer) for regularizer in flat_regularizer]):
      raise ValueError("Not all the passed regularizers are callable objects.")

  # Produce names for the variables. In the case of a tuple or nested tuple,
  # this is just a sequence of numbers, but for a flat `namedtuple`, we use
  # the field names. NOTE: this could be extended to nested `namedtuple`s,
  # but for now that's extra complexity that's not used anywhere.
  name_prefix = name or "initial_state"
  try:
    name_suffixes = [
        state_size._fields[i] for i in xrange(len(flat_state_size))]
  except (AttributeError, IndexError):
    name_suffixes = range(len(flat_state_size))

  flat_initial_state = []

  for name_suffix, size, init, regularizer in zip(
      name_suffixes, flat_state_size, flat_initializer, flat_regularizer):
    shape_with_batch_dim = [1] + tf.TensorShape(size).as_list()

    variable_name = "{}_{}".format(name_prefix, name_suffix)
    initial_state_module = basic.TrainableVariable(
        shape_with_batch_dim, dtype=dtype, initializers={"w": init},
        regularizers={"w": regularizer}, name=variable_name)
    initial_state_variable = initial_state_module()

    tiled_name = "state_{}_tiled".format(name_suffix)

    initial_state_variable_dims = initial_state_variable.get_shape().ndims
    tile_dims = [batch_size] + [1] * (initial_state_variable_dims - 1)
    flat_initial_state.append(
        tf.tile(initial_state_variable, tile_dims, name=tiled_name))

  return nest.pack_sequence_as(structure=state_size,
                               flat_sequence=flat_initial_state)


@six.add_metaclass(abc.ABCMeta)
class RNNCore(base.AbstractModule):
  """Superclass for Recurrent Neural Network Cores.

  This class defines the basic functionality that every core should implement,
  mainly the `initial_state` method which will return an example of their
  initial state.
  It also inherits from the two interfaces it should be compatible with, which
  are `snt.Module` and `tf.contrib.rnn.RNNCell`.

  As with any other `snt.Module` any subclass must implement a `_build` method
  that constructs the graph that corresponds to a core. Such a `_build` method
  should always have the same interface, which is the following:

      output, next_state = self._build(input, prev_state)

  where output, next_state, input, and prev_state are arbitrarily nested
  tensors. Such structures can be defined according to the following
  grammar:

      element = tuple(element*) | list(element*) | tf.Tensor

  This class is to be used with tensorflow containers such as `rnn` in
  tensorflow.python.ops.rnn. These containers only accept
  `tf.contrib.rnn.RNNCell` objects, hence the need to comply with its interface.
  This way, all the RNNCores should expose a `state_size` and `output_size`
  properties.
  """
  __metaclass__ = abc.ABCMeta

  def _initial_state_scope(self, name):
    """Defines the name scope of the initial_state ops."""
    return name if name else "%s_initial_state" % self.scope_name

  def initial_state(self, batch_size, dtype=tf.float32, trainable=False,
                    trainable_initializers=None, trainable_regularizers=None,
                    name=None, **unused_kwargs):
    """Builds the default start state for an RNNCore.

    Args:
      batch_size: An int, or scalar int32 Tensor representing the batch size.
      dtype: The data type to use for the state.
      trainable: Boolean that indicates whether to learn the initial state.
        Note that intializers and regularizers will be ignored if
        `trainable=False`.
      trainable_initializers: An initializer function or nested structure of
          functions with same structure as the `state_size` property of the
          core, to be used as initializers of the initial state variable.
      trainable_regularizers: Optional regularizer function or nested structure
        of functions with the same structure as the `state_size` property of the
        core, to be used as regularizers of the initial state variable. As a
        default, no regularizers are used. A regularizer should be a function
        that takes a single `Tensor` as an input and returns a scalar `Tensor`
        output, e.g. the L1 and L2 regularizers in `tf.contrib.layers`.
      name: Optional string used to prefix the initial state variable names, in
          the case of a trainable initial state. If not provided, defaults to
          the name of the module.

    Returns:
      A tensor or nested tuple of tensors with same structure and shape as the
      `state_size` property of the core.

    Raises:
      ValueError: if the user passes initializers that are not functions.
      ValueError: if the user passes regularizers that are not functions.
    """
    with tf.name_scope(self._initial_state_scope(name)):
      if not trainable:
        return self.zero_state(batch_size, dtype)
      else:
        return trainable_initial_state(
            batch_size, self.state_size, dtype,
            initializers=trainable_initializers,
            regularizers=trainable_regularizers,
            name=self._initial_state_scope(name))

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.

    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
      the shapes `[batch_size x s]` for each s in `state_size`.
    """
    # Keep scope for backwards compatibility.
    with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return rnn_cell_impl._zero_state_tensors(  # pylint: disable=protected-access
          self.state_size, batch_size, dtype)


class TrainableInitialState(base.AbstractModule):
  """Helper Module that creates a learnable initial state for an RNNCore.

  This class receives an example (possibly nested) initial state of an RNNCore,
  and returns a state that has the same shape, structure, and values, but is
  trainable. Additionally, the user may specify a boolean mask that
  indicates which parts of the initial state should be trainable.

  This allows users to train an unrolled RNNCore with a learnable initial state
  in the following way:

      core = ... # Any RNNCore module object.
      initial_state = core.initial_state(batch_size, dtype)
      trainable_initial_state = snt.TrainableInitialState(initial_state)()
      output, final_state = tf.nn.dynamic_rnn(
          core, input_sequence, initial_state=trainable_initial_state)
  """

  def __init__(self, initial_state, mask=None, name="trainable_initial_state"):
    """Constructs the Module that introduces a trainable state in the graph.

    It receives an initial state that will be used as the initial values for the
    trainable variables that the module contains, and optionally a mask that
    indicates the parts of the initial state that should be learnable.

    Args:
      initial_state: tensor or arbitrarily nested iterables of tensors.
      mask: optional boolean mask. It should have the same nested structure as
       the given initial_state.
      name: module name.

    Raises:
      TypeError: if mask is not a list of booleans or None.
    """
    super(TrainableInitialState, self).__init__(name=name)

    # Since python 2.7, DeprecationWarning is ignored by default.
    # Turn on the warning:
    warnings.simplefilter("always", DeprecationWarning)
    warnings.warn("Use the trainable flag in initial_state instead.",
                  DeprecationWarning, stacklevel=2)

    if mask is not None:
      flat_mask = nest.flatten(mask)
      if not all([isinstance(m, bool) for m in flat_mask]):
        raise TypeError("Mask should be None or a list of boolean values.")
      nest.assert_same_structure(initial_state, mask)

    self._mask = mask
    self._initial_state = initial_state

  def _build(self):
    """Connects the module to the graph.

    Returns:
      The learnable state, which has the same type, structure and shape as
        the `initial_state` passed to the constructor.
    """
    flat_initial_state = nest.flatten(self._initial_state)
    if self._mask is not None:
      flat_mask = nest.flatten(self._mask)
      flat_learnable_state = [
          _single_learnable_state(state, state_id=i, learnable=mask)
          for i, (state, mask) in enumerate(zip(flat_initial_state, flat_mask))]
    else:
      flat_learnable_state = [_single_learnable_state(state, state_id=i)
                              for i, state in enumerate(flat_initial_state)]

    return nest.pack_sequence_as(structure=self._initial_state,
                                 flat_sequence=flat_learnable_state)


class RNNCellWrapper(RNNCore):
  """RNN core that delegates to a `tf.contrib.rnn.RNNCell`."""

  def __init__(self, cell_ctor, *args, **kwargs):
    """Constructs the cell, within this module's variable scope.

    Args:
      cell_ctor: Callable that instantiates a `tf.contrib.rnn.RNNCell`.
      *args: Arguments to pass to `cell_ctor`.
      **kwargs: Keyword arguments to pass to `cell_ctor`.
        If `name` is provided, it is passed to `RNNCore.__init__` as well.
        If `custom_getter` is provided, it is passed to `RNNCore.__init__`
        but not to `cell_ctor`.
    """
    super(RNNCellWrapper, self).__init__(
        name=kwargs.get("name"),
        custom_getter=kwargs.pop("custom_getter", None))

    with self._enter_variable_scope():
      self._cell = cell_ctor(*args, **kwargs)

  def _build(self, inputs, prev_state):
    return self._cell(inputs, prev_state)

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size


def with_doc(fn_with_doc_to_copy):
  """Returns a decorator to copy documentation from the given function.

  Docstring is copied, including *args and **kwargs documentation.

  Args:
    fn_with_doc_to_copy: Function whose docstring, including *args and
      **kwargs documentation, is to be copied.

  Returns:
    Decorated version of `wrapper_init` with documentation copied from
    `fn_with_doc_to_copy`.
  """

  def decorator(wrapper_init):
    # Wrap the target class's constructor (to assume its docstring),
    # but invoke the wrapper class's constructor.
    @wrapt.decorator
    def wrapping_fn(unused_wrapped, instance, args, kwargs):
      wrapper_init(instance, *args, **kwargs)
    return wrapping_fn(fn_with_doc_to_copy)  # pylint: disable=no-value-for-parameter

  return decorator


def wrap_rnn_cell_class(wrapped_class):
  """Wraps an RNN cell class with a sub-class of `RNNCellWrapper`.

  The returned wrapper class will contain an `__init__` method whose
  docstring, *args, and **kwargs are based on `wrapped_class.__init__`.

  Args:
    wrapped_class: A sub-class (NOT an instance) of `tf.contrib.rnn.RNNCell`.

  Returns:
    A sub-class (NOT an instance) of `RNNCellWrapper`, with an `__init__`
    method that delegates to that of `wrapped_class`.
  """

  class Wrapper(RNNCellWrapper):

    @with_doc(wrapped_class.__init__)
    def __init__(self, *args, **kwargs):
      super(Wrapper, self).__init__(wrapped_class, *args, **kwargs)

  return Wrapper
