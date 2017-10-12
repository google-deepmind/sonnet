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
"""Stop gradient custom getter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def stop_gradient(getter, *args, **kwargs):
  """Custom getter which prevents variables being optimized.

  Usage like:

    with tf.variable_scope("", custom_getter=snt.custom_getters.stop_gradient):
        net = snt.Linear(num_hidden)(net)

  or, using the `custom_getter` constructor argument,

    linear = snt.Linear(num_hidden,
                        custom_getter=snt.custom_getters.stop_gradient)
    net = linear(net)

  will result in the gradient with respect to the variables in the linear
  module being `None`. By default, the variables will still be in the trainable
  variables collection.

  When used with a Sonnet module, the module must be constructed inside the
  variable scope with the custom getter. Just building the module inside said
  variable scope will not use the custom getter.

  Args:
    getter: The true getter to call.
    *args: Arguments, in the same format as tf.get_variable.
    **kwargs: Keyword arguments, in the same format as tf.get_variable.

  Returns:
    The return value of `getter(*args, **kwargs)` with a tf.stop_gradient.
  """

  return tf.stop_gradient(getter(*args, **kwargs))
