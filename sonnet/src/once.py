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
"""Utility to run functions and methods once."""

import uuid

from sonnet.src import utils

_ONCE_PROPERTY = "_snt_once"


def _check_no_output(output):
  if output is not None:
    raise ValueError("@snt.once decorated functions cannot return values")


def once(f):
  """Decorator which ensures a wrapped method is only ever run once.

      >>> @snt.once
      ... def f():
      ...   print('Hello, world!')
      >>> f()
      Hello, world!
      >>> f()
      >>> f()

  If `f` is a method then it will be evaluated once per instance:

      >>> class MyObject:
      ...   @snt.once
      ...   def f(self):
      ...     print('Hello, world!')

      >>> o = MyObject()
      >>> o.f()
      Hello, world!
      >>> o.f()

      >>> o2 = MyObject()
      >>> o2.f()
      Hello, world!
      >>> o.f()
      >>> o2.f()

  If an error is raised during execution of `f` it will be raised to the user.
  Next time the method is run, it will be treated as not having run before.

  Args:
    f: A function to wrap which should only be called once.

  Returns:
    Wrapped version of `f` which will only evaluate `f` the first time it is
    called.
  """

  # TODO(tomhennigan) Perhaps some more human friendly identifier?
  once_id = uuid.uuid4()

  @utils.decorator
  def wrapper(wrapped, instance, args, kwargs):
    """Decorator which ensures a wrapped method is only ever run once."""
    if instance is None:
      # NOTE: We can't use the weakset since you can't weakref None.
      if not wrapper.seen_none:
        _check_no_output(wrapped(*args, **kwargs))
        wrapper.seen_none = True
      return

    # Get or set the `seen` set for this object.
    seen = getattr(instance, _ONCE_PROPERTY, None)
    if seen is None:
      seen = set()
      setattr(instance, _ONCE_PROPERTY, seen)

    if once_id not in seen:
      _check_no_output(wrapped(*args, **kwargs))
      seen.add(once_id)

  wrapper.seen_none = False

  decorated = wrapper(f)  # pylint: disable=no-value-for-parameter,assignment-from-none
  decorated.__snt_once_wrapped__ = f
  return decorated
