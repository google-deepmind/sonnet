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
"""Context manager to switch a custom getter on or off."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Context(object):
  """Contextually switching a custom getter on.

  Example usage, once Sonnet modules accept a custom_getter argument:

    custom_getter = snt.custom_getters.Context(snt.custom_getters.stop_gradient)
    lin = snt.Linear(10, custom_getter=custom_getter)

    lin(net1)  # custom getter not used, gradients on
    with custom_getter:
      lin(net2)  # custom getter used, gradients off


  Warning: If the custom getter affects the way the variable is created, then
  switching it on or off after the variable has been created will have no
  effect. For example, it is not possible to selectively switch off
  trainability using `custom_getters.non_trainable`, since this is a
  creation-time attribute. It is however possible to selectively switch
  off gradients using `custom_getters.stop_gradient`, since
  this applies an operation to the variable.
  """

  def __init__(self, getter):
    """Initializes a contextual switch for a custom getter.

    Args:
      getter: The custom getter which we may want to switch on.

    Returns:
      A custom getter which can also be used as a context manager.
      Entering the context enables the custom getter.
    """
    self._count = 0
    self._getter = getter

  def __call__(self, getter, *args, **kwargs):
    if self._count:
      return self._getter(getter, *args, **kwargs)
    else:
      return getter(*args, **kwargs)

  def __enter__(self):
    self._count += 1

  def __exit__(self, exception_type, exception_value, exception_traceback):
    self._count -= 1
