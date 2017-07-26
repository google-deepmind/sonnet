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

"""Base class for TensorFlow snt.

This file contains the Abstract Base Class for defining Modules in TensorFlow.
A Module is an object that can be connected into the Graph multiple times
using the __call__ method, sharing variables automatically with no need to
explicitly use scopes or specify reuse=True.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

# Dependency imports
import six
from sonnet.python.modules import util
import tensorflow as tf


class Error(Exception):
  """Base class for all errors from snt.

  This is thrown to indicate a Neural Network specific problem, e.g. wrong
  module arity, module is not connected to the graph when it should be,
  tried to wire together incompatible modules, etc.
  """


class NotConnectedError(Error):
  """Error raised when operating on a module that has not yet been connected.

  Some module properties / methods are valid to access before the module has
  been connected into the graph, but some are not. This Error is raised when
  the user attempts to do anything not valid before connection.
  """


class ParentNotBuiltError(Error):
  """Error raised when the parent of a module has not been built yet.

  For example, when making a transpose of modules that inherit from
  `module.Transposable`, the parent has to be connected to the graph before the
  child transpose to ensure that shape inference has already occurred.
  """


class IncompatibleShapeError(Error):
  """Error raised when the shape of the input at build time is incompatible."""


class UnderspecifiedError(Error):
  """Error raised when too little information is available.

  This does not typically mean the user is trying to do something that doesn't
  work (in which case `IncompatibleShapeError` should be used), just that
  some more information needs to be provided in order to build the Graph.
  """


class NotSupportedError(Error):
  """Error raised when something that cannot be supported is requested.

  For example a Dilated Convolution module cannot be transposed.
  """


class NotInitializedError(Error):
  """Error raised when connecting an uninitialized Sonnet module.

  Before they can be connected, all Sonnet modules must call
  `AbstractModule.__init__` (e.g. via a `super` call).
  """


class DifferentGraphError(Error):
  """Error raised when trying to connect a Sonnet module to multiple Graphs."""


SubgraphInputs = collections.namedtuple("SubgraphInputs", ("args", "kwargs"))


ConnectedSubGraph = collections.namedtuple(
    "ConnectedSubGraph", ("builder", "name_scope", "inputs", "outputs"))


@six.add_metaclass(abc.ABCMeta)
class AbstractModule(object):
  """Superclass for Sonnet Modules.

  This class defines the functionality that every module should implement,
  principally the `build` method which is wrapped using `tf.make_template`
  and called from `__call__`. Every time the module is called it will
  be connected into the graph but using the same shared set of variables, thanks
  to the template.

  For this to work correctly, the `build` implementation in the derived class
  must access all variables using `tf.get_variable`, not `tf.Variable`. The same
  set of variables must be created each time, if this is not the case an Error
  will be raised.

  Every subclass must call this class' `__init__` at the start of their
  `__init__`, passing the relevant name. If this step is omitted variable
  sharing will not work.
  """

  def __init__(self, _sentinel=None, custom_getter=None,
               name=None):  # pylint: disable=invalid-name
    """Performs the initialisation necessary for all AbstractModule instances.

    Every subclass of AbstractModule must begin their constructor with a call to
    this constructor, i.e. `super(MySubModule, self).__init__(name=name)`.

    If you instantiate sub-modules in __init__ you must create them within the
    `_enter_variable_scope` context manager to ensure they are in the module's
    variable scope. Alternatively, instantiate sub-modules in `_build`.

    Args:
      _sentinel: Variable that only carries a non-None value if `__init__` was
          called without named parameters. If this is the case, a deprecation
          warning is issued in form of a `ValueError`.
      custom_getter: Callable or dictionary of callables to use as
        custom getters inside the module. If a dictionary, the keys
        correspond to regexes to match variable names. See the `tf.get_variable`
        documentation for information about the custom_getter API.
      name: Name of this module. Used to construct the Templated build function.
          If `None` the module's class name is used (converted to snake case).

    Raises:
      TypeError: If `name` is not a string.
      TypeError: If a given `custom_getter` is not callable.
      ValueError: If `__init__` was called without named arguments.
    """
    if _sentinel is not None:
      raise ValueError("Calling AbstractModule.__init__ without named "
                       "arguments is deprecated.")

    if name is None:
      name = util.to_snake_case(self.__class__.__name__)
    elif not isinstance(name, six.string_types):
      raise TypeError("Name must be a string.")

    self._connected_subgraphs = []

    # If the given custom getter is a dictionary with a per-variable custom
    # getter, wrap it into a single custom getter.
    if isinstance(custom_getter, collections.Mapping):
      self._custom_getter = util._custom_getter_router(  # pylint: disable=protected-access
          custom_getter_map=custom_getter,
          name_fn=lambda name: name[len(self.scope_name) + 1:])
    else:
      if not (custom_getter is None or callable(custom_getter)):
        raise TypeError("Given custom_getter is not callable.")
      self._custom_getter = custom_getter

    self._template = tf.make_template(name,
                                      self._build_wrapper,
                                      create_scope_now_=True,
                                      custom_getter_=self._custom_getter)

    self._original_name = name
    self._unique_name = self._template.variable_scope.name.split("/")[-1]

    # Update __call__ and the object docstrings to enable better introspection
    self.__doc__ = self._build.__doc__
    self.__call__.__func__.__doc__ = self._build.__doc__

    # Keep track of which graph this module has been connected to. Sonnet
    # modules cannot be connected to multiple graphs, as transparent variable
    # sharing is impossible in that case.
    self._graph = None

  def _build_wrapper(self, *args, **kwargs):
    """Function which will be wrapped in a Template to do variable sharing.

    Passes through all arguments to the _build method, and returns the
    corresponding outputs, plus the name_scope generated by this call of the
    template.

    Args:
      *args: args list for self._build
      **kwargs: kwargs dict for self._build

    Returns:
      A tuple containing (output from _build, scope_name).
    """
    output = self._build(*args, **kwargs)
    # Make a dummy subscope to check the name scope we are in. We could read
    # the name scope from one of the outputs produced, except that the outputs
    # could have been produced from a subscope instantiated by the build
    # function, for example if inner modules are present. Calling name_scope
    # here and creating a new subscope guarantees we get the right answer.
    # Because we don't create an ops inside this dummy scope, no extra memory
    # will be consumed.
    with tf.name_scope("dummy") as scope_name:
      this_scope_name = scope_name[:-len("/dummy/")]
    return output, this_scope_name

  def _check_init_called(self):
    """Checks that the base class's __init__ method has been called.

    Raises:
      NotInitializedError: `AbstractModule.__init__` has not been called.
    """
    try:
      self._template
    except AttributeError:
      raise NotInitializedError("You may have forgotten to call super at the "
                                "start of %s.__init__."
                                % self.__class__.__name__)

  def _check_same_graph(self):
    """Checks that the module is not being connect to multiple Graphs.

    An instance of a Sonnet module 'owns' the variables it contains, and permits
    seamless variable sharing. As such, connecting a single module instance to
    multiple Graphs is not possible - this function will raise an error should
    that occur.

    Raises:
      DifferentGraphError: if the module is connected to a different Graph than
        it was previously used in.
    """
    current_graph = tf.get_default_graph()
    if self._graph is None:
      self._graph = current_graph
    elif self._graph != current_graph:
      raise DifferentGraphError("Cannot connect module to multiple Graphs.")

  @abc.abstractmethod
  def _build(self, *args, **kwargs):
    """Add elements to the Graph, computing output Tensors from input Tensors.

    Subclasses must implement this method, which will be wrapped in a Template.

    Args:
      *args: Input Tensors.
      **kwargs: Additional Python flags controlling connection.

    Returns:
      output Tensor(s).
    """

  def __call__(self, *args, **kwargs):
    """Operator overload for calling.

    This is the entry point when users connect a Module into the Graph. The
    underlying _build method will have been wrapped in a Template by the
    constructor, and we call this template with the provided inputs here.

    Args:
      *args: Arguments for underlying _build method.
      **kwargs: Keyword arguments for underlying _build method.

    Returns:
      The result of the underlying _build method.
    """
    self._check_init_called()
    self._check_same_graph()
    outputs, this_name_scope = self._template(*args, **kwargs)
    # Connect the module only if self._template returns with no errors.
    inputs = SubgraphInputs(args, kwargs)
    self._connected_subgraphs.append(
        ConnectedSubGraph(self, this_name_scope, inputs, outputs))
    return outputs

  @property
  def name_scopes(self):
    """Returns a tuple of all name_scopes generated by this module."""
    return tuple(subgraph.name_scope for subgraph in self._connected_subgraphs)

  @property
  def variable_scope(self):
    """Returns the variable_scope declared by the module.

    It is valid for library users to access the internal templated
    variable_scope, but only makes sense to do so after connection. Therefore we
    raise an error here if the variable_scope is requested before connection.

    The only case where it does make sense to access the variable_scope before
    connection is to get the post-uniquification name, which we support using
    the separate .name property.

    Returns:
      variable_scope: `tf.VariableScope` instance of the internal `tf.Template`.

    Raises:
      NotConnectedError: If the module is not connected to the Graph.
    """
    self._ensure_is_connected()
    return self._template.variable_scope

  @property
  def scope_name(self):
    """Returns the full name of the Module's variable scope."""
    return self._template.variable_scope.name

  @property
  def module_name(self):
    """Returns the name of the Module."""
    return self._unique_name

  @property
  def is_connected(self):
    """Returns true iff the Module been connected to the Graph at least once."""
    return bool(self._connected_subgraphs)

  @property
  def connected_subgraphs(self):
    """Returns the subgraphs created by this module so far."""
    return tuple(self._connected_subgraphs)

  @property
  def last_connected_subgraph(self):
    """Returns the last subgraph created by this module.

    Returns:
      The last connected subgraph.

    Raises:
      NotConnectedError: If the module is not connected to the Graph.
    """
    self._ensure_is_connected()
    return self._connected_subgraphs[-1]

  @classmethod
  def get_possible_initializer_keys(cls):
    """Returns the keys the dictionary of variable initializers may contain.

    This provides the user with a way of knowing the initializer keys that are
    available without having to instantiate a sonnet module. Subclasses may
    override this class method if they need additional arguments to determine
    what initializer keys may be provided.

    Returns:
      Set with strings corresponding to the strings that may be passed to the
          constructor.
    """
    return getattr(cls, "POSSIBLE_INITIALIZER_KEYS", set())

  def _ensure_is_connected(self):
    """Raise an Error if the module has not been connected yet.

    Until the module is connected into the Graph, any variables created do
    not exist yet and cannot be created in advance due to not knowing the size
    of the input Tensor(s). This assertion ensures that any variables contained
    in this module must now exist.

    Raises:
      NotConnectedError: If the module is not connected to the Graph.
    """
    if not self.is_connected:
      raise NotConnectedError(
          "Variables in {} not instantiated yet, __call__ the module "
          "first.".format(self.scope_name))

  def _enter_variable_scope(self, reuse=None):
    """Returns a contextlib.contextmanager to enter the internal variable scope.

    This is useful for situations where submodules must be declared in the
    constructor, or somewhere else that is not called under the `_build` method.
    If such a case arises, calling `with self._enter_variable_scope():` will
    cause the variables in the submodule to be correctly scoped.

    An example justification for this is to allow the `Transposable` interface
    to be implemented - you might want to construct all the submodules at
    construction time so that you can call `.transpose()` and connect the
    result of that before connecting the non-transposed module.

    ```python
    class SomeModule(snt.AbstractModule):
      def __init__(self, name="some_module"):
        super(SomeModule, self).__init__(name=name)
        with self._enter_variable_scope():
          # We need to construct this submodule before we get to the _build
          # method, for some reason.
          self._sub_mod = snt.SomeSubmodule(name="some_submodule")

      def _build(self, input):
        # Connect to the already constructed submodule.
        return self._sub_mod(input)
    ```

    If you omit this then the submodule and parent module will appear to
    be "side by side" rather than nested when viewed in the Graph viewer, and
    functions such as `snt.get_variables_in_module()` or the `get_variables()`
    method will not know about variables defined in the submodule.

    Args:
      reuse: Boolean passed to `tf.variable_scope`.

    Returns:
      `contextlib.contextmanager` of the variable_scope inside the template.
    """
    self._check_init_called()
    self._check_same_graph()
    return tf.variable_scope(self._template.variable_scope, reuse=reuse)

  def get_variables(self, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
    """Returns tuple of `tf.Variable`s declared inside this module.

    Note that this operates by searching this module's variable scope,
    and so does not know about any modules that were constructed elsewhere but
    used inside this module.

    Args:
      collection: Collection to restrict query to. By default this is
        `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
        variables such as moving averages.

    Returns:
      A tuple of `tf.Variable` objects.

    Raises:
      NotConnectedError: If the module is not connected to the Graph.
    """
    return util.get_variables_in_scope(
        self.variable_scope, collection=collection)

  def __getstate__(self):
    raise NotSupportedError(
        "Sonnet AbstractModule instances cannot be serialized. You should "
        "instead serialize all necessary configuration which will allow "
        "modules to be rebuilt.")


@six.add_metaclass(abc.ABCMeta)
class Transposable(object):
  """Transposable module interface.

    The Transposable interface requires that transposable modules implement
    a method called `transpose`, returning a module that is the transposed
    version of the one the method is called on.
    Calling the method twice should return a module with the same specifications
    as the original module.

    When implementing a transposable module, special care is required to make
    sure that parameters needed to instantiate the module are provided as
    functions whose invocation is deferred to graph construction time.

    For example, in Linear we might want to call:

    ```python
    linear = snt.Linear(name="linear", output_size=output_size)
    linear_transpose = linear.transpose()
    ```

    where the output_size for linear_transpose is not known yet, as linear is
    not yet connected to the graph: output_size is passed to linear_transpose's
    constructor as a lambda returning linear.input_size. The lambda will return
    the correct value once linear is given an input.
    Notice that linear_transpose's output_size value does not need to be defined
    until the module is connected to the graph.
  """

  @abc.abstractmethod
  def transpose(self, name=None, **kwargs):
    """Builds and returns transposed version of module.

    Args:
      name: Name of the transposed module.
      **kwargs: Additional Python flags controlling transposition.

    Returns:
      Transposed version of the module.
    """

  @abc.abstractmethod
  def input_shape(self):
    """Returns shape of input `Tensor` passed at last call to `build`."""


class Module(AbstractModule):
  """Module wrapping a function provided by the user."""

  def __init__(self, build, name=None):
    """Constructs a module with a given build function.

    The Module class can be used to wrap a function assembling a network into a
    module.

    For example, the following code implements a simple one-hidden-layer MLP
    model by defining a function called make_model and using a Module instance
    to wrap it.

    ```python
    def make_model(inputs):
      lin1 = snt.Linear(name="lin1", output_size=10)(inputs)
      relu1 = tf.nn.relu(lin1, name="relu1")
      lin2 = snt.Linear(name="lin2", output_size=20)(relu1)
      return lin2

    model = snt.Module(name='simple_mlp', build=make_model)
    outputs = model(inputs)
    ```

    The `partial` package from `functools` can be used to bake configuration
    parameters into the function at construction time, as shown in the following
    example.

    ```python
    from functools import partial

    def make_model(inputs, output_sizes):
      lin1 = snt.Linear(name="lin1", output_size=output_sizes[0])(inputs)
      relu1 = tf.nn.relu(lin1, name="relu1")
      lin2 = snt.Linear(name="lin2", output_size=output_sizes[1])(relu1)
      return lin2

    model = snt.Module(name='simple_mlp',
                       build=partial(make_model, output_size=[10, 20])
    outputs = model(inputs)
    ```

    Args:
      build: Callable to be invoked when connecting the module to the graph.
          The `build` function is invoked when the module is called, and its
          role is to specify how to add elements to the Graph, and how to
          compute output Tensors from input Tensors.
          The `build` function signature can include the following parameters:
            *args - Input Tensors.
            **kwargs - Additional Python parameters controlling connection.
      name: Module name. If set to `None` (the default), the name will be set to
          that of the `build` callable converted to `snake_case`. If `build` has
          no name, the name will be 'module'.

    Raises:
      TypeError: If build is not callable.
    """
    if not callable(build):
      raise TypeError("Input 'build' must be callable.")
    if name is None:
      name = util.name_for_callable(build)
    super(Module, self).__init__(name=name)
    self._build_function = build

  def _build(self, *args, **kwargs):
    """Forwards call to the passed-in build function."""
    return self._build_function(*args, **kwargs)
