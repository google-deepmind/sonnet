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
"""Enables module construction to be deferred."""

from sonnet.src import base


class Deferred(base.Module):
  """Defers the construction of another module until the first call.

  Deferred can be used to declare modules that depend on computed properties of
  other modules before those modules are defined. This allows users to separate
  the declaration and use of modules. For example at the start of your program
  you can declare two modules which are coupled:

      >>> encoder = snt.Linear(64)
      >>> decoder = snt.Deferred(lambda: snt.Linear(encoder.input_size))

  Later you can use these naturally (note: that using `decoder` first would
  cause an error since `encoder.input_size` is only defined after `encoder` has
  been called):

      >>> x = tf.ones([8, 32])
      >>> y = encoder(x)
      >>> z = decoder(y)  # Constructs the Linear encoder by calling the lambda.

  The result will satisfy the following conditions:

      >>> assert x.shape == z.shape
      >>> assert y.shape == [8, 64]
      >>> assert decoder.input_size == encoder.output_size
      >>> assert decoder.output_size == encoder.input_size
  """

  def __init__(self, constructor, call_methods=("__call__",), name=None):
    """Initializes the `Deferred` module.

    Args:
      constructor: A no argument callable which constructs the module to defer
        to. The first time one of the `call_methods` are called the constructor
        will be run and then the constructed module will be called with the same
        method and arguments as the deferred module.
      call_methods: Methods which should trigger construction of the target
        module. The default value configures this module to construct the first
        time `__call__` is run. If you want to add methods other than call you
        should explicitly pass them (optionally), for example
        `call_methods=("__call__", "encode", "decode")`.
      name: Name for the deferred module.
    """
    super().__init__(name=name)
    self._constructor = constructor
    self._target = None

    for call_method in call_methods:
      if call_method == "__call__":
        # Has to be handled separately because __call__ cannot be overridden at
        # the instance level.
        # See: https://docs.python.org/3/reference/datamodel.html#special-lookup
        continue

      setattr(self, call_method, _materialize_then_call(self, call_method))

  @property
  @base.no_name_scope
  def target(self):
    """Returns the target module.

    If the constructor has not already run this will trigger construction.
    Subsequent calls to `target` will return the same instance.

    Returns:
      A `Module` instance as created by `self.constructor()` .
    """
    if self._target is None:
      self._target = self._constructor()
      self._constructor = None
    return self._target

  @base.no_name_scope
  def __call__(self, *args, **kwargs):
    return self.target(*args, **kwargs)  # pylint: disable=not-callable

  def __str__(self):
    return "Deferred({})".format(str(self.target))

  def __repr__(self):
    return "Deferred({})".format(repr(self.target))

  def __getattr__(self, name):
    if name != "_target" and hasattr(self, "_target"):
      if self._target is not None:
        return getattr(self._target, name)

    raise AttributeError("'%s' object has no attribute '%s'" %
                         (self.__class__.__name__, name))

  def __setattr__(self, name, value):
    if name != "_target" and hasattr(self, "_target"):
      if self._target is not None:
        setattr(self._target, name, value)
        return

    super().__setattr__(name, value)

  def __delattr__(self, name):
    if name != "_target" and hasattr(self, "_target"):
      if self._target is not None:
        return delattr(self._target, name)

    super().__delattr__(name)


def _materialize_then_call(module, method_name):

  def wrapped(*args, **kwargs):
    return getattr(module.target, method_name)(*args, **kwargs)

  return wrapped
