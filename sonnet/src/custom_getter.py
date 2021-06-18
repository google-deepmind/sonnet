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
"""Custom getter for module members."""

from typing import Any, Callable, ContextManager, Iterable, Optional, Type

import contextlib
from sonnet.src import base
import tensorflow as tf
import tree

_DEFAULT_CLASSES = [base.Module]


@contextlib.contextmanager
def _patch_getattribute(cls, new_getattribute):
  orig_getattribute = cls.__getattribute__  # pytype: disable=attribute-error
  cls.__getattribute__ = new_getattribute
  try:
    yield
  finally:
    cls.__getattribute__ = orig_getattribute


def _custom_getter(
    getter: Callable[[Any], Any],
    classes: Optional[Iterable[Type[Any]]] = None,
    instances: Optional[Iterable[Any]] = None) -> ContextManager[Any]:
  """Applies the given `getter` when getting members of given `classes`.

  For example:
  >>> class X:
  ...   values = [1, 2]

  >>> x = X()
  >>> x.values
  [1, 2]

  >>> with _custom_getter(lambda x: x + [3], classes=[X]):
  ...   x.values
  [1, 2, 3]

  >>> with _custom_getter(lambda x: x + [3], instances={x}):
  ...   x.values
  [1, 2, 3]

  >>> x.values
  [1, 2]

  Args:
    getter: A callable to apply to each element of the class members.
    classes: The classes in which the getter is applied. If `None`, defaults to
      `set(o.__class__ for o in instances)`. If `classes and `instances` are
      both `None`, defaults to `[Module]`.
    instances: The instances in which the getter is applied. If `None`, the
      getter will apply in all instances of `classes`.

  Returns:
    A context manager in which the custom getter is active.
  """
  # Workaround for the fact that we can't annotate the type as `Collection` in
  # Python < 3.6.
  if instances is not None:
    instances = frozenset(instances)

  if classes is None:
    if instances is None:
      classes = _DEFAULT_CLASSES
    else:
      classes = frozenset(o.__class__ for o in instances)

  stack = contextlib.ExitStack()

  for cls in classes:
    orig_getattribute = cls.__getattribute__  # pytype: disable=attribute-error

    def new_getattribute(obj, name, orig_getattribute=orig_getattribute):
      attr = orig_getattribute(obj, name)

      if (instances is None) or (obj in instances):
        return getter(attr)
      else:
        return attr

    stack.enter_context(_patch_getattribute(cls, new_getattribute))

  return stack


def custom_variable_getter(
    getter: Callable[[tf.Variable], Any],
    classes: Optional[Iterable[Type[Any]]] = None,
    instances: Optional[Iterable[Any]] = None) -> ContextManager[Any]:
  """Applies the given `getter` when getting variables of given `classes`.

  If a member is a nested structure containing any variable, `getter` will be
  applied to each variable in the nest.

  For example:
  >>> class Times2(snt.Module):
  ...   def __init__(self):
  ...     super(Times2, self).__init__()
  ...     self.v = tf.Variable(2.)
  ...
  ...   def __call__(self, x):
  ...     return x * self.v

  >>> x = 42.
  >>> times2 = Times2()

  >>> with tf.GradientTape() as tape:
  ...   y = times2(x)
  >>> assert tape.gradient(y, times2.v).numpy() == x

  >>> with custom_variable_getter(tf.stop_gradient):
  ...   with tf.GradientTape() as tape:
  ...     y = times2(x)
  >>> assert tape.gradient(y, times2.v) is None

  >>> with tf.GradientTape() as tape:
  ...   y = times2(x)
  >>> assert tape.gradient(y, times2.v).numpy() == x

  Args:
    getter: A callable to apply to each variable of the class.
    classes: The classes in which the getter is applied. If `None`, defaults to
      `set(o.__class__ for o in instances)`. If `classes and `instances` are
      both `None`, defaults to `[Module]`.
    instances: The instances in which the getter is applied. If `None`, the
      getter will apply in all instances of `classes`.

  Returns:
    A context manager in which the custom getter is active.
  """

  def wrapped_getter(x):
    x_flat = tree.flatten(x)
    if any(_is_variable(it) for it in x_flat):
      return tree.unflatten_as(
          x, [getter(it) if _is_variable(it) else it for it in x_flat])
    else:
      return x

  return _custom_getter(wrapped_getter, classes=classes, instances=instances)


def _is_variable(x):
  return isinstance(x, tf.Variable)
