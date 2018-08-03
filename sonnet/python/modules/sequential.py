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

"""Sequential Module for TensorFlow snt.

A Module that wraps a list of other modules and ops, connecting the output of
each to the input of the next.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from sonnet.python.modules import base
import tensorflow as tf


class Sequential(base.AbstractModule):
  """Builds a module out of a sequence of callables.

  Note that `Sequential` is limited in the range of possible architectures
  it can handle. This is a deliberate design decision; `Sequential` is only
  meant to be used for the simple case of fusing together modules/ops where
  the input of a particular module/op is the output of the previous one. Another
  restriction is that it is not possible to have extra arguments in the `_build`
  method that are passed to the constituents of the module - for example,
  if there is a `BatchNorm` module in `Sequential` and the user wishes to switch
  the `is_training` flag. If this is the desired use case, the recommended
  solution is to use `snt.Module` to wrap a custom function, as shown in the
  following example:

  https://github.com/deepmind/sonnet/blob/master/sonnet/examples/module_with_build_args.py
  """

  def __init__(self, layers, name="sequential"):
    """Constructs a Sequential module.

    This feeds the output of each layer into the next and returns the output
    of the final layer.

    If a layer returns a tuple, it is assumed that this must be unpacked into
    the argument list of the next layer. If it is not a tuple, it is simply
    passed through to the next layer unchanged.

    Args:
      layers: Iterable of callables to stack together, which can be modules
          or ops.
      name: Name of the module.

    Raises:
      TypeError: If `layers` is None or contains any non-callable items.
    """
    super(Sequential, self).__init__(name=name)

    # Store a copy of the iterable in a tuple to ensure users cannot modify the
    # iterable later, and protect against iterables which can only be read once.
    self._layers = tuple(layers)

    is_not_callable = [(i, mod) for i, mod in enumerate(self._layers)
                       if not callable(mod)]

    if is_not_callable:
      raise TypeError("Items {} not callable with types: {}".format(
          ", ".join(str(i) for i, _ in is_not_callable),
          ", ".join(type(layer).__name__ for _, layer in is_not_callable)))

  def _build(self, *args):
    """Connects the Sequential module into the graph.

    Args:
      *args: A tuple of inputs, to be unpacked as the arguments to the first
          layer.

    Returns:
      The output value of the last layer.
    """
    net = args

    if not self._layers:
      # If the sequential is passed a single arg, this will end up being
      # wrapped in an extra layer of tuple by *args. Normally we internally
      # handle this in the loop below, but if there are no layers we unpack here
      # in order to make Sequential([]) act like an identity, which seems right.
      if len(args) == 1:
        return args[0]
      else:
        return args

    for layer in self._layers:
      if isinstance(net, tuple):
        net = layer(*net)
      else:
        net = layer(net)

    return net

  @property
  def layers(self):
    return self._layers

  def get_variables(self, *args, **kwargs):
    """Provide a warning that get_variables on Sequential always returns ()."""
    tf.logging.warning(
        "Calling Sequential.get_variables, which will always return an empty "
        "tuple. get_variables() can only return variables created directly by "
        "a Module, or created by submodules directly created inside the "
        "Module. Sequential is constructed from already constructed submodules "
        "and so this will always be empty. See the documentation for more "
        "details, but tl;dr if you need to connect some modules sequentially "
        "and call get_variables on the result, writing a simple custom module "
        "is the simplest way. Another option is to call get_all_variables().")
    return super(Sequential, self).get_variables(*args, **kwargs)
