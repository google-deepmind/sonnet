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
# =============================================================================
"""Module for experimental sonnet functions and classes.

This file contains functions and classes that are being tested until they're
either removed or promoted into the wider sonnet library.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import weakref
import tensorflow as tf


def reuse_vars(method):
  """Wraps an arbitrary method so it does variable sharing.

  The wrapper creates variables the first time it calls `method` for a given
  `(tf.Graph, tf.VariableScope)` pair and reuses them for subsequent calls
  (provided they use the same variable scope). These variables are created in
  the context of a variable scope provided to the wrapper as an instance
  attribute.

  The variables defined in two wrapped method calls are shared if
  and only if all of the following hold:

  1. They refer to the same method.
  2. The two calls use the same variable scope.
  3. All the ops in `method` live in the same `tf.Graph`, which is set as the
     default graph.

  The variable scope can be provided either as a keyword argument
  `variable_scope` or as an instance attribute. These two options are mutually
  exclusive.

  To use an instance attribute as a variable scope the following must be true:

  1. `method` is an instance method.
  2. The calling instance has an attribute `variable_scope` of type
     `tf.VariableScope`.

  Args:
    method: The method to wrap.

  Returns:
    The wrapped method.
  """
  initialized = weakref.WeakSet()

  # Ensure that the argument passed in is really a method by checking that the
  # first positional argument to it is "self".
  arg_spec = inspect.getargspec(method)
  is_method = arg_spec.args and arg_spec.args[0] == "self"

  if not is_method:
    raise TypeError("reuse_vars can only be used with methods.")

  @functools.wraps(method)
  def wrapper(*args, **kwargs):
    """Calls `method` with a variable scope whose reuse flag is set correctly.

    The first time the wrapper is called it creates a
    `(tf.Graph, tf.VariableScope)` key and checks it for membership in
    `initialized`. The check is `False` if and only if this is the first time
    the wrapper has been called with the key, otherwise the check is `True`. The
    result of this check is used as the `reuse` flag for entering the provided
    variable scope before calling `method`.

    Here are two examples of how to use the reuse_vars decorator.

    1. Decorate an arbitrary instance method with a `variable_scope` attribute:

      ```python
      class Reusable(object):

        def __init__(self, name):
          with tf.variable_scope(name) as vs:
            self.variable_scope = vs

        @snt.experimental.reuse_vars
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

        @snt.experimental.reuse_vars
        def add_a(self, input_tensor):
          a = tf.get_variable("a", shape=input_tensor.get_shape())
          return a + input_tensor

        # We don't need @snt.experimental.reuse_vars here because build is
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
      *args: The positional arguments (Tensors) passed to the wrapped method.
      **kwargs: The keyword arguments passed to the wrapped method.

    Returns:
      Output of the wrapped method.

    Raises:
      ValueError: If no variable scope is provided or if `method` is a method
                  and a variable_scope keyword argument is also provided.
    """
    obj = args[0]

    def default_context_manager(reuse=None):
      variable_scope = obj.variable_scope
      return tf.variable_scope(variable_scope, reuse=reuse)

    variable_scope_context_manager = getattr(obj, "_enter_variable_scope",
                                             default_context_manager)

    initialized_key = obj

    reuse = initialized_key in initialized

    with variable_scope_context_manager(reuse=reuse):
      with tf.name_scope(method.__name__):
        out_ops = method(*args, **kwargs)
        initialized.add(initialized_key)
        return out_ops

  return wrapper
