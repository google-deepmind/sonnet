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

"""Utility functions for dealing with Sonnet Modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import inspect
import re
import weakref

# Dependency imports
import six
import tensorflow as tf
import wrapt

from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as variable_scope_ops


def get_variable_scope_name(value):
  """Returns the name of the variable scope indicated by the given value.

  Args:
    value: String, variable scope, or object with `variable_scope` attribute
    (e.g., Sonnet module).

  Returns:
    The name (a string) of the corresponding variable scope.

  Raises:
    ValueError: If `value` does not identify a variable scope.
  """
  # If the object has a "variable_scope" property, use it.
  value = getattr(value, "variable_scope", value)
  if isinstance(value, tf.VariableScope):
    return value.name
  elif isinstance(value, six.string_types):
    return value
  else:
    raise ValueError("Not a variable scope: {}".format(value))


def get_variables_in_scope(scope, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
  """Returns a tuple `tf.Variable`s in a scope for a given collection.

  Args:
    scope: `tf.VariableScope` or string to retrieve variables from.
    collection: Collection to restrict query to. By default this is
        `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
        variables such as moving averages.

  Returns:
    A tuple of `tf.Variable` objects.
  """
  scope_name = get_variable_scope_name(scope)

  if scope_name:
    # Escape the name in case it contains any "." characters. Add a closing
    # slash so we will not search any scopes that have this scope name as a
    # prefix.
    scope_name = re.escape(scope_name) + "/"

  return tuple(tf.get_collection(collection, scope_name))


def get_variables_in_module(module,
                            collection=tf.GraphKeys.TRAINABLE_VARIABLES):
  """Returns tuple of `tf.Variable`s declared inside an `snt.Module`.

  Note that this operates by searching the variable scope a module contains,
  and so does not know about any modules which were constructed elsewhere but
  used inside this module.

  Args:
    module: `snt.Module` instance to query the scope of.
    collection: Collection to restrict query to. By default this is
      `tf.Graphkeys.TRAINABLE_VARIABLES`, which doesn't include non-trainable
      variables such as moving averages.

  Returns:
    A tuple of `tf.Variable` objects.

  Raises:
    NotConnectedError: If the module is not connected to the Graph.
  """
  return module.get_variables(collection=collection)


def _check_nested_callables(dictionary, object_name):
  """Checks if all items in the dictionary and in subdictionaries are callables.

  Args:
    dictionary: Dictionary of callables or other dictionaries with callables.
    object_name: The name of the object that is expected in the dictionary.
      E.g. 'Initializer', 'Partitioner' or 'Regularizer'. The first letter
      should be capitalised as this will be the first word in the error message.

  Raises:
    TypeError: If the dictionary contains something that is not either a
      dictionary or a callable.
  """
  for key, entry in six.iteritems(dictionary):
    if hasattr(entry, "items"):
      _check_nested_callables(entry, object_name)
    elif not callable(entry):
      raise TypeError(
          "{} for '{}' is not a callable function or dictionary"
          .format(object_name, key))


def _assert_is_dictlike(maybe_dictlike, valid_keys):
  """Raises a TypeError iff `maybe_dictlike` is not a dictlike object."""
  # This covers a common mistake when people use incorrect dictionary nesting
  # for initializers / partitioners etc. The previous error message was quite
  # opaque, this should be much clearer.
  if not hasattr(maybe_dictlike, "__getitem__"):
    raise TypeError(
        "Expected a dict-like object with possible keys %s, received %s" %
        (str(valid_keys), str(maybe_dictlike)))


def check_initializers(initializers, keys):
  """Checks the given initializers.

  This checks that `initializers` is a dictionary that only contains keys in
  `keys`, and furthermore the entries in `initializers` are functions or
  further dictionaries (the latter used, for example, in passing initializers
  to modules inside modules) that must satisfy the same constraints.

  Args:
    initializers: Dictionary of initializers (allowing nested dictionaries) or
      None.
    keys: Iterable of valid keys for `initializers`.

  Returns:
    Copy of checked dictionary of initializers. If `initializers=None`, an empty
    dictionary will be returned.

  Raises:
    KeyError: If an initializer is provided for a key not in `keys`.
    TypeError: If a provided initializer is not a callable function, or
      `initializers` is not a Mapping.
  """
  if initializers is None:
    return {}
  _assert_is_dictlike(initializers, valid_keys=keys)

  keys = set(keys)

  if not set(initializers) <= keys:
    extra_keys = set(initializers) - keys
    raise KeyError(
        "Invalid initializer keys {}, initializers can only "
        "be provided for {}".format(
            ", ".join("'{}'".format(key) for key in extra_keys),
            ", ".join("'{}'".format(key) for key in keys)))

  _check_nested_callables(initializers, "Initializer")

  return dict(initializers)


def check_partitioners(partitioners, keys):
  """Checks the given partitioners.

  This checks that `partitioners` is a dictionary that only contains keys in
  `keys`, and furthermore the entries in `partitioners` are functions or
  further dictionaries (the latter used, for example, in passing partitioners
  to modules inside modules) that must satisfy the same constraints.

  Args:
    partitioners: Dictionary of partitioners (allowing nested dictionaries) or
        None.
    keys: Iterable of valid keys for `partitioners`.

  Returns:
    Checked dictionary of partitioners. If `partitioners=None`, an empty
    dictionary will be returned.

  Raises:
    KeyError: If an partitioner is provided for a key not in `keys`.
    TypeError: If a provided partitioner is not a callable function, or
      `partitioners` is not a Mapping.
  """
  if partitioners is None:
    return {}
  _assert_is_dictlike(partitioners, valid_keys=keys)

  keys = set(keys)

  if not set(partitioners) <= keys:
    extra_keys = set(partitioners) - keys
    raise KeyError(
        "Invalid partitioner keys {}, partitioners can only "
        "be provided for {}".format(
            ", ".join("'{}'".format(key) for key in extra_keys),
            ", ".join("'{}'".format(key) for key in keys)))

  _check_nested_callables(partitioners, "Partitioner")

  return partitioners


def check_regularizers(regularizers, keys):
  """Checks the given regularizers.

  This checks that `regularizers` is a dictionary that only contains keys in
  `keys`, and furthermore the entries in `regularizers` are functions or
  further dictionaries (the latter used, for example, in passing regularizers
  to modules inside modules) that must satisfy the same constraints.

  Args:
    regularizers: Dictionary of regularizers (allowing nested dictionaries) or
      None.
    keys: Iterable of valid keys for `regularizers`.

  Returns:
    Copy of checked dictionary of regularizers. If `regularizers=None`, an empty
    dictionary will be returned.

  Raises:
    KeyError: If an regularizers is provided for a key not in `keys`.
    TypeError: If a provided regularizer is not a callable function, or
      `regularizers` is not a Mapping.
  """
  if regularizers is None:
    return {}
  _assert_is_dictlike(regularizers, valid_keys=keys)

  keys = set(keys)

  if not set(regularizers) <= keys:
    extra_keys = set(regularizers) - keys
    raise KeyError(
        "Invalid regularizer keys {}, regularizers can only "
        "be provided for {}".format(
            ", ".join("'{}'".format(key) for key in extra_keys),
            ", ".join("'{}'".format(key) for key in keys)))

  _check_nested_callables(regularizers, "Regularizer")

  return dict(regularizers)


def _is_scope_prefix(scope_name, prefix_name):
  """Checks that `prefix_name` is a proper scope prefix of `scope_name`."""

  if not prefix_name:
    return True

  if not scope_name.endswith("/"):
    scope_name += "/"

  if not prefix_name.endswith("/"):
    prefix_name += "/"

  return scope_name.startswith(prefix_name)


# pylint: disable=protected-access
def _get_sliced_variables(var_list):
  """Separates the sliced (partitioned) and unsliced variables in var_list.

  Args:
    var_list: a list of variables.

  Returns:
    A list of unsliced variables in var_list, and a dict mapping names to parts
    for the sliced variables in var_list.
  """
  unsliced_variables = []
  sliced_variables = collections.defaultdict(lambda: [])
  for var in var_list:
    if var._save_slice_info:
      sliced_variables[var._save_slice_info.full_name].append(var)
    else:
      unsliced_variables.append(var)
  return unsliced_variables, sliced_variables
# pylint: enable=protected-access


def custom_getter_router(custom_getter_map, name_fn):
  """Creates a custom getter than matches requests to dict of custom getters.

  Custom getters are callables which implement the
  [custom getter API]
  (https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/get_variable).

  The returned custom getter dispatches calls based on pattern matching the
  name of the requested variable to the keys of custom_getter_map. For example,

      {
        ".*/w": snt.custom_getters.stop_gradient,
      }

  will match all variables named with the suffix "/w". The `name_fn` is
  provided to allow processing of the name, such as stripping off a scope prefix
  before matching.

  Args:
    custom_getter_map: Mapping of regular expressions to custom getter
      functions.
    name_fn: Callable to map variable name through before matching to regular
      expressions. This might, for example, strip off a scope prefix.

  Returns:
    A custom getter.

  Raises:
    TypeError: If an entry in `custom_getter_map` is not a callable function.
  """

  for custom_getter in custom_getter_map.values():
    if not callable(custom_getter):
      raise TypeError("Given custom_getter is not callable.")

  def _custom_getter(getter, name, *args, **kwargs):
    """A custom getter that routes based on pattern matching the variable name.

    Args:
      getter: The true getter to call.
      name: The fully qualified variable name, i.e. including all scopes.
      *args: Arguments, in the same format as tf.get_variable.
      **kwargs: Keyword arguments, in the same format as tf.get_variable.

    Returns:
      The return value of the appropriate custom getter. If there are no
      matches, it returns the return value of `getter`.

    Raises:
      KeyError: If more than one pattern matches the variable name.
    """
    bare_name = name_fn(name)
    matches = [
        (custom_getter, pattern)
        for pattern, custom_getter in custom_getter_map.items()
        if re.match(pattern, bare_name) is not None]

    num_matches = len(matches)

    if num_matches == 0:
      return getter(name, *args, **kwargs)
    elif num_matches == 1:
      custom_getter, pattern = matches[0]
      return custom_getter(getter, name, *args, **kwargs)
    else:
      raise KeyError("More than one custom_getter matched {} ({}): {}".format(
          name, bare_name, [pattern for _, pattern in matches]))

  return _custom_getter


def get_normalized_variable_map(scope_or_module,
                                collection=tf.GraphKeys.GLOBAL_VARIABLES,
                                context=None,
                                group_sliced_variables=True):
  """Builds map of `tf.Variable`s in scope or module with normalized names.

  The names of the variables are normalized to remove the scope prefix.

  Args:
    scope_or_module: Scope or module to build map from.
    collection: Collection to restrict query to. By default this is
        `tf.Graphkeys.GLOBAL_VARIABLES`, which includes non-trainable variables
        such as moving averages.
    context: Scope or module, identical to or parent of `scope`. If given, this
        will be used as the stripped prefix. By default `None`, which means
        `context=scope`.
    group_sliced_variables: Boolean, if set to True, sliced variables are
       grouped together in the returned map; if set to False, each partition of
       a sliced variable is a separate (key, value) pair.

  Returns:
    Dictionary mapping normalized variable name to `tf.Variable`, or a list
        of `tf.Variables` if the variable is a sliced (partitioned) variable.

  Raises:
    ValueError: If `context` is given but is not a proper prefix of `scope`.
  """
  scope_name = get_variable_scope_name(scope_or_module)

  if context is None:
    context = scope_or_module

  prefix = get_variable_scope_name(context)
  prefix_length = len(prefix) + 1 if prefix else 0

  if not _is_scope_prefix(scope_name, prefix):
    raise ValueError("Scope '{}' is not prefixed by '{}'.".format(
        scope_name, prefix))

  variables = get_variables_in_scope(scope_name, collection)

  if not group_sliced_variables:
    single_vars = variables
    grouped_vars = dict()
  else:
    single_vars, grouped_vars = _get_sliced_variables(variables)

  var_map = {var.op.name[prefix_length:]: var for var in single_vars}
  for full_name, var_group in grouped_vars.items():
    name = full_name[prefix_length:]
    if name in var_map:
      raise ValueError("Mixing slices and non-slices with the same name: " +
                       str(name))
    var_map[name] = var_group
  return var_map


def get_saver(scope, collections=(tf.GraphKeys.GLOBAL_VARIABLES,),  # pylint: disable=redefined-outer-name
              context=None, **kwargs):
  """Builds a `tf.train.Saver` for the scope or module, with normalized names.

  The names of the variables are normalized to remove the scope prefix.
  This allows the same variables to be restored into another similar scope or
  module using a complementary `tf.train.Saver` object.

  Args:
    scope: Scope or module. Variables within will be saved or restored.
    collections: Sequence of collections of variables to restrict
        `tf.train.Saver` to. By default this is `tf.GraphKeys.GLOBAL_VARIABLES`
        which includes moving averages variables as well as trainable variables.
    context: Scope or module, identical to or parent of `scope`. If given, this
        will be used as the stripped prefix.
    **kwargs: Extra keyword arguments to pass to tf.train.Saver.

  Returns:
    A `tf.train.Saver` object for Variables in the scope or module.
  """

  variable_map = {}
  for collection in collections:
    variable_map.update(get_normalized_variable_map(scope, collection, context))

  return tf.train.Saver(var_list=variable_map, **kwargs)


def has_variable_scope(obj):
  """Determines whether the given object has a variable scope."""
  return "variable_scope" in dir(obj)


def _format_table(rows, join_lines=True):
  format_str = ""
  for col in range(len(rows[0])):
    column_width = max(len(row[col]) for row in rows)
    format_str += "{:<" + str(column_width) + "}  "

  output_rows = (format_str.format(*row).strip() for row in rows)
  return "\n".join(output_rows) if join_lines else output_rows


def variable_map_items(variable_map):
  """Yields an iterator over (string, variable) pairs in the variable map.

  In general, variable maps map variable names to either a `tf.Variable`, or
  list of `tf.Variable`s (in case of sliced variables).

  Args:
    variable_map: dict, variable map over which to iterate.

  Yields:
    (string, tf.Variable) pairs.
  """
  for key, var_or_vars in six.iteritems(variable_map):
    if isinstance(var_or_vars, (list, tuple)):
      for variable in var_or_vars:
        yield key, variable
    else:
      yield key, var_or_vars


def _get_vars_to_collections(variables):
  """Returns a dict mapping variables to the collections they appear in."""
  var_to_collections = collections.defaultdict(lambda: [])
  if isinstance(variables, dict):
    variables = list(v for _, v in variable_map_items(variables))
  for graph in set(v.graph for v in variables):
    for collection_name in list(graph.collections):
      entries = set(entry for entry in graph.get_collection(collection_name)
                    if isinstance(entry, tf.Variable))
      # For legacy reasons, tf.GraphKeys.GLOBAL_VARIABLES == "variables".
      # Correcting for this here, to avoid confusion.
      if collection_name == tf.GraphKeys.GLOBAL_VARIABLES:
        collection_name = "global_variables"
      for var in entries.intersection(variables):
        var_to_collections[var].append(collection_name)
  return var_to_collections


def format_variables(variables, join_lines=True):
  """Takes a collection of variables and formats it as a table."""
  rows = []
  rows.append(("Variable", "Shape", "Type", "Collections", "Device"))
  var_to_collections = _get_vars_to_collections(variables)
  for var in sorted(variables, key=lambda var: var.op.name):
    if var.get_shape().is_fully_defined():
      shape = "x".join(str(dim) for dim in var.get_shape().as_list())
    else:
      shape = "undefined"
    dtype = repr(var.dtype.base_dtype).replace("tf.", "")
    coll = ", ".join(sorted(var_to_collections[var]))
    rows.append((var.op.name, shape, dtype, coll, var.device))
  return _format_table(rows, join_lines)


def format_variable_map(variable_map, join_lines=True):
  """Takes a key-to-variable map and formats it as a table."""
  rows = []
  rows.append(("Key", "Variable", "Shape", "Type", "Collections", "Device"))
  var_to_collections = _get_vars_to_collections(variable_map)

  sort_key = lambda item: (item[0], item[1].name)
  for key, var in sorted(variable_map_items(variable_map), key=sort_key):
    shape = "x".join(str(dim) for dim in var.get_shape().as_list())
    dtype = repr(var.dtype.base_dtype).replace("tf.", "")
    coll = ", ".join(sorted(var_to_collections[var]))
    rows.append((key, var.op.name, shape, dtype, coll, var.device))
  return _format_table(rows, join_lines)


def log_variables(variables=None):
  """Logs variable information.

  This function logs the name, shape, type, collections, and device for either
  all variables or a given iterable of variables.

  Args:
    variables: iterable of variables; if not provided, then all variables
        (in the default graph) are logged.
  """
  if variables is None:
    variables = tf.global_variables() + tf.local_variables()
  for row in format_variables(variables, join_lines=False):
    tf.logging.info(row)


def _num_bytes_to_human_readable(num_bytes):
  """Returns human readable string of how much memory `num_bytes` fills."""
  if num_bytes < (2 ** 10):
    return "%d B" % num_bytes
  elif num_bytes < (2 ** 20):
    return "%.3f KB" % (float(num_bytes) / (2 ** 10))
  elif num_bytes < (2 ** 30):
    return "%.3f MB" % (float(num_bytes) / (2 ** 20))
  else:
    return "%.3f GB" % (float(num_bytes) / (2 ** 30))


def summarize_variables(variables=None):
  """Logs a summary of variable information.

  This function groups Variables by dtype and prints out the number of Variables
  and the total number of scalar values for each datatype, as well as the total
  memory consumed.

  For Variables of type tf.string, the memory usage cannot be accurately
  calculated from the Graph as the memory requirements change based on what
  strings are actually stored, which can only be determined inside a session.
  In this case, the amount of memory used to stored the pointers to the strings
  is logged, along with a warning.

  Args:
    variables: iterable of variables; if not provided, then all variables
      (in the default graph) are summarized.
  """

  variable_counts = count_variables_by_type(variables=variables)
  total_num_scalars = 0
  total_num_bytes = 0

  # Sort by string representation of type name, so output is deterministic.
  for dtype in sorted(variable_counts,
                      key=lambda dtype: "%r" % dtype):
    var_info_for_type = variable_counts[dtype]
    num_bytes = var_info_for_type["num_scalars"] * dtype.size
    total_num_scalars += var_info_for_type["num_scalars"]
    total_num_bytes += num_bytes
    tf.logging.info("%r: %d variables comprising %d scalars, %s",
                    dtype, var_info_for_type["num_variables"],
                    var_info_for_type["num_scalars"],
                    _num_bytes_to_human_readable(num_bytes))


def count_variables_by_type(variables=None):
  """Returns a dict mapping dtypes to number of variables and scalars.

  Args:
    variables: iterable of `tf.Variable`s, or None. If None is passed, then all
      global and local variables in the current graph are used.

  Returns:
    A dict mapping tf.dtype keys to a dict containing the keys 'num_scalars' and
      'num_variables'.
  """
  if variables is None:
    variables = tf.global_variables() + tf.local_variables()
  unique_types = set(v.dtype.base_dtype for v in variables)
  results_dict = {}
  for dtype in unique_types:
    if dtype == tf.string:
      tf.logging.warning(
          "NB: string Variables present. The memory usage for these  Variables "
          "will not be accurately computed as it depends on the exact strings "
          "stored in a particular session.")
    vars_of_type = [v for v in variables if v.dtype.base_dtype == dtype]
    num_scalars = sum(v.shape.num_elements() for v in vars_of_type)
    results_dict[dtype] = {
        "num_variables": len(vars_of_type),
        "num_scalars": num_scalars
    }
  return results_dict


def reuse_variables(method):
  """Wraps an arbitrary method so it does variable sharing.

  This decorator creates variables the first time it calls `method`, and reuses
  them for subsequent calls. The object that calls `method` provides a
  `tf.VariableScope`, either as a `variable_scope` attribute or as the return
  value of an `_enter_variable_scope()` method.

  The first time the wrapped method is invoked, it enters the caller's
  `tf.VariableScope` with `reuse=False`. On all subsequent calls it enters the
  same variable scope with `reuse=True`.

  Variables are created in the context of the `tf.VariableScope` provided by the
  caller object. Ops are created with an additional `tf.name_scope()`, which
  adds a scope for the wrapped method name. For example:

  ```python
  class MyClass(object):

    def __init__(self, name):
      with tf.variable_scope(None, default_name=name) as variable_scope:
        self.variable_scope = variable_scope

    @snt.reuse_variables
    def add_x(self, tensor):
      x = tf.get_variable("x", shape=tensor.get_shape())
      return tensor + x

  module = MyClass("my_module_name")
  input_tensor = tf.zeros(shape=(5,))

  # This creates the variable "my_module_name/x"
  # and op "my_module_name/add_x/add"
  output = module.add_x(input_tensor)
  ```

  For performance when executing eagerly it may be desirable to additionally
  annotate these methods using `defun`, such that they are encapsulated as
  graph functions. This is not recommended if your method returns a variable
  since the output of `defun` would be an op that returned the variable's value
  when evaluated (rather than the variable instance).

  ```python
  class FooModule(snt.AbstractModule):
    def _build(self, inputs):
      return complex_math(inputs)

    @tfe.defun
    @snt.reuse_variables
    def more_complex_stuff(self, inputs):
      return more_complex_math(inputs)
  ```

  Args:
    method: The method to wrap.

  Returns:
    The wrapped method.
  """
  initialized_variable_scopes_eager = set()
  initialized_variable_scopes_graph = weakref.WeakKeyDictionary()

  # Ensure that the argument passed in is really a method by checking that the
  # first positional argument to it is "self".
  arg_spec = inspect.getargspec(method)
  is_method = arg_spec.args and arg_spec.args[0] == "self"

  if not is_method:
    raise TypeError("reuse_variables can only be used with methods.")

  @wrapt.decorator
  def eager_test(method, obj, args, kwargs):
    """Validates runtime state in eager mode."""
    # If @reuse_variables is combined with @property, obj is passed in args
    # and method is still unbound at this stage.
    if obj is None:
      obj = args[0]

    if tf.executing_eagerly() and not hasattr(obj, "_template"):
      raise ValueError(
          "reuse_variables is not supported in eager mode except in Sonnet "
          "modules.")

    return method(*args, **kwargs)

  @wrapt.decorator
  def call_method(method, obj, args, kwargs):
    """Calls `method` with a variable scope whose reuse flag is set correctly.

    The first time the wrapper is called it creates a
    `(tf.Graph, tf.VariableScope)` key and checks it for membership in
    `initialized_variable_scopes`. The check is `False` if and only if this is
    the first time the wrapper has been called with the key, otherwise the
    check is `True`. The result of this check is used as the `reuse` flag for
    entering the provided variable scope before calling `method`.

    Here are two examples of how to use the reuse_variables decorator.

    1. Decorate an arbitrary instance method with a `variable_scope` attribute:

      ```python
      class Reusable(object):

        def __init__(self, name):
          with tf.variable_scope(None, default_name=name) as vs:
            self.variable_scope = vs

        @snt.reuse_variables
        def add_a(self, input_tensor):
          a = tf.get_variable("a", shape=input_tensor.get_shape())
          return a + input_tensor

      obj = Reusable("reusable")
      x = tf.constant(5.0)
      out1 = obj.add_a(x)
      out2 = obj.add_a(x)
      # out1 == out2
      ```

    2. Decorating a snt.AbstractModule instance method:

      ```python
      class ReusableModule(snt.AbstractModule):

        @snt.reuse_variables
        def add_a(self, input_tensor):
          a = tf.get_variable("a", shape=input_tensor.get_shape())
          return a + input_tensor

        # We don't need @snt.reuse_variables here because build is
        wrapped by # `tf.make_template` inside `snt.AbstractModule`.
        def _build(self, input_tensor):
          b = tf.get_variable("b", shape=input_tensor.get_shape())
          return b + self.add_a(input_tensor)

      obj = Reusable("reusable")
      x = tf.constant(5.0)
      out1 = obj(x)
      out2 = obj(x)
      # out1 == out2
      ```

    Args:
      method: The method to wrap.
      obj: The object instance passed to the wrapped method.
      args: The positional arguments (Tensors) passed to the wrapped method.
      kwargs: The keyword arguments passed to the wrapped method.

    Returns:
      Output of the wrapped method.

    Raises:
      ValueError: If no variable scope is provided or if `method` is a method
                  and a variable_scope keyword argument is also provided.
    """

    # If @reuse_variables is combined with @property, obj is passed in args
    # and method is still unbound at this stage.
    if obj is None:
      obj = args[0]

    def default_context_manager(reuse=None):
      variable_scope = obj.variable_scope
      return tf.variable_scope(variable_scope, reuse=reuse)

    variable_scope_context_manager = getattr(obj, "_enter_variable_scope",
                                             default_context_manager)

    with ops.init_scope():
      # We need `init_scope` incase we're running inside a defun. In that case
      # what we want is information about where the function will be called not
      # where the function is being built.
      graph = tf.get_default_graph()
      will_call_in_eager_context = tf.executing_eagerly()

    if will_call_in_eager_context:
      initialized_variable_scopes = initialized_variable_scopes_eager
    else:
      if graph not in initialized_variable_scopes_graph:
        initialized_variable_scopes_graph[graph] = set()
      initialized_variable_scopes = initialized_variable_scopes_graph[graph]

    # Temporarily enter the variable scope to capture it
    with variable_scope_context_manager() as tmp_variable_scope:
      variable_scope = tmp_variable_scope

    reuse = variable_scope.name in initialized_variable_scopes

    # Enter the pure variable scope with reuse correctly set
    with variable_scope_ops._pure_variable_scope(  # pylint:disable=protected-access
        variable_scope, reuse=reuse) as pure_variable_scope:
      current_name_scope = tf.get_default_graph().get_name_scope()
      # Force tf.name_scope to treat current_name_scope as an "absolute" scope
      # so we can re-enter it.
      if current_name_scope and current_name_scope[-1] != "/":
        current_name_scope += "/"
      with tf.name_scope(current_name_scope):
        module_name = pure_variable_scope.name
        method_name = to_snake_case(method.__name__)
        method_name_scope = "{}/{}".format(module_name, method_name)
        with tf.name_scope(method_name_scope) as scope:
          if hasattr(obj, "_capture_variables"):
            with obj._capture_variables():  # pylint: disable=protected-access
              out_ops = method(*args, **kwargs)
          else:
            out_ops = method(*args, **kwargs)
      initialized_variable_scopes.add(pure_variable_scope.name)
      try:
        # If `obj` is a Sonnet module, let it know it's been connected
        # to the TF graph.
        obj._is_connected = True  # pylint: disable=protected-access
        if not tf.executing_eagerly():
          obj._add_connected_subgraph(  # pylint: disable=protected-access
              method, out_ops, scope, *args, **kwargs)
      except AttributeError:
        pass
    return out_ops

  return eager_test(call_method(method))  # pylint: disable=no-value-for-parameter


def name_for_callable(func):
  """Returns a module name for a callable or `None` if no name can be found."""
  if isinstance(func, functools.partial):
    return name_for_callable(func.func)

  try:
    name = func.__name__
  except AttributeError:
    return None

  if name == "<lambda>":
    return None
  else:
    return to_snake_case(name)


def to_snake_case(camel_case):
  """Returns a CamelCase string as a snake_case string."""
  if not re.match(r"^[A-Za-z_]\w*$", camel_case):
    raise ValueError(
        "Input string %s is not a valid Python identifier." % camel_case)

  # Add underscore at word start and ends.
  underscored = re.sub(r"([A-Z][a-z])", r"_\1", camel_case)
  underscored = re.sub(r"([a-z])([A-Z])", r"\1_\2", underscored)
  # Add underscore before alphanumeric chunks.
  underscored = re.sub(r"([a-z])([0-9][^_]*)", r"\1_\2", underscored)
  # Remove any underscores at start or end of name and convert to lowercase.
  return underscored.strip("_").lower()


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
  """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.

  Args:
    x: A `Tensor`.

  Returns:
    The input `Tensor`.
  """
  return x


def sort_by_name(variables):
  """Returns a tuple of `variables` sorted ascending by name."""
  return tuple(sorted(variables, key=lambda v: v.name))


@contextlib.contextmanager
def notify_about_variables(callback):
  """Calls `callback(var)` for all `tf.{Variable,get_variable}` results.

  Callback should not modify the variable passed in. Use cases that require
  variables to be modified should use `variable_creator_scope` directly and sit
  within the variable creator stack.

  >>> variables = []
  >>> with notify_about_variables(variables.append):
  ...   v = tf.Variable(1.0, name='v')
  ...   w = tf.get_variable('w', [])
  >>> assert variables == [v, w]

  Args:
    callback: a callable taking a single argument which is a tf.Variable.

  Yields:
    `None` - used for contextmanager API.
  """
  def _tracking_creator(getter, **kwargs):
    v = getter(**kwargs)
    callback(v)
    return v

  with variable_scope_ops.variable_creator_scope(_tracking_creator):
    yield
