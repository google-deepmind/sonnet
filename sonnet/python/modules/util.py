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

import re

# Dependency imports
import six
import tensorflow as tf


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
  if isinstance(scope, tf.VariableScope):
    scope = scope.name

  # Escape the name in case it contains any "." characters. Add a closing slash
  # so we will not search any scopes that have this scope name as a prefix.
  scope_name = re.escape(scope) + "/"

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
    if isinstance(entry, dict):
      _check_nested_callables(entry, object_name)
    elif not callable(entry):
      raise TypeError(
          "{} for '{}' is not a callable function or dictionary"
          .format(object_name, key))


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
    TypeError: If a provided initializer is not a callable function, or if the
      dict of initializers is not in fact a dict.
  """
  if initializers is None:
    return {}

  keys = set(keys)

  # If the user is creating modules that nests other modules, then it is
  # possible that they might not nest the initializer dictionaries correctly. If
  # that is the case, then we might find that initializers is not a dict here.
  # We raise a helpful exception in this case.
  if not issubclass(type(initializers), dict):
    raise TypeError("A dict of initializers was expected, but not "
                    "given. You should double-check that you've nested the "
                    "initializers for any sub-modules correctly.")

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
    TypeError: If a provided partitioner is not a callable function.
  """
  if partitioners is None:
    return {}

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
    TypeError: If a provided regularizers is not a callable function, or if the
      dict of regularizers is not in fact a dict.
  """
  if regularizers is None:
    return {}

  keys = set(keys)

  # If the user is creating modules that nests other modules, then it is
  # possible that they might not nest the regularizer dictionaries correctly. If
  # that is the case, then we might find that regularizers is not a dict here.
  # We raise a helpful exception in this case.
  if not issubclass(type(regularizers), dict):
    raise TypeError("A dict of regularizers was expected, but not "
                    "given. You should double-check that you've nested the "
                    "regularizers for any sub-modules correctly.")

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

  if not scope_name.endswith("/"):
    scope_name += "/"

  if not prefix_name.endswith("/"):
    prefix_name += "/"

  return scope_name.startswith(prefix_name)


def get_normalized_variable_map(scope_or_module,
                                collection=tf.GraphKeys.GLOBAL_VARIABLES,
                                context=None):
  """Builds map of `tf.Variable`s in scope or module with normalized names.

  The names of the variables are normalized to remove the scope prefix.

  Args:
    scope_or_module: Scope or module to build map from.
    collection: Collection to restrict query to. By default this is
        `tf.Graphkeys.VARIABLES`, which includes non-trainable variables such
        as moving averages.
    context: Scope or module, identical to or parent of `scope`. If given, this
        will be used as the stripped prefix. By default `None`, which means
        `context=scope`.

  Returns:
    Dictionary mapping normalized variable name to `tf.Variable`.

  Raises:
    ValueError: If `context` is given but is not a proper prefix of `scope`.
  """
  scope = getattr(scope_or_module, "variable_scope", scope_or_module)

  if context is None:
    context = scope
  context_scope = getattr(context, "variable_scope", context)

  scope_name = scope.name
  prefix = context_scope.name
  if not _is_scope_prefix(scope_name, prefix):
    raise ValueError("Scope '{}' is not prefixed by '{}'.".format(
        scope_name, prefix))

  prefix_length = len(prefix) + 1

  variables = get_variables_in_scope(scope, collection)

  return {variable.name[prefix_length:]: variable for variable in variables}


def get_saver(scope, collections=(tf.GraphKeys.GLOBAL_VARIABLES,),
              context=None):
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

  Returns:
    A `tf.train.Saver` object for Variables in the scope or module.
  """

  variable_map = {}

  for collection in collections:
    variable_map.update(get_normalized_variable_map(scope, collection, context))

  return tf.train.Saver(var_list=variable_map)


def has_variable_scope(obj):
  """Determines whether the given object has a variable scope."""
  return "variable_scope" in dir(obj)


def _format_table(rows):
  format_str = ""
  for col in range(len(rows[0])):
    column_width = max(len(row[col]) for row in rows)
    format_str += "{:<" + str(column_width) + "}  "

  return "\n".join(format_str.format(*row).strip() for row in rows)


def format_variables(variables):
  """Takes a collection of variables and formats it as a table."""
  rows = []
  rows.append(("Variable", "Shape", "Type"))
  for var in sorted(variables, key=lambda var: var.name):
    shape = "x".join(str(dim) for dim in var.get_shape().as_list())
    dtype = repr(var.dtype.base_dtype)
    rows.append((var.name, shape, dtype))
  return _format_table(rows)


def format_variable_map(variable_map):
  """Takes a key-to-variable map and formats it as a table."""
  rows = []
  rows.append(("Key", "Variable", "Shape", "Type"))
  for key in sorted(variable_map.keys()):
    var = variable_map[key]
    shape = "x".join(str(dim) for dim in var.get_shape().as_list())
    dtype = repr(var.dtype.base_dtype)
    rows.append((key, var.name, shape, dtype))
  return _format_table(rows)
