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
"""Custom getter to override specific named arguments of get_variable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six


def override_args(**kwargs):
  """Creates a custom getter that applies specified named arguments.

  Args:
    **kwargs: Overriding arguments for the custom getter to use in preference
      the named arguments it's called with.

  Returns:
    Custom getter.
  """

  override_kwargs = kwargs

  def custom_getter(getter, *args, **kwargs):
    """Custom getter with certain named arguments overridden.

    Args:
      getter: Underlying variable getter to invoke.
      *args: Arguments, compatible with those of tf.get_variable.
      **kwargs: Keyword arguments, compatible with those of tf.get_variable.

    Returns:
      The result of invoking `getter(*args, **kwargs)` except that certain
      kwargs entries may have been overridden.
    """
    kwargs.update(override_kwargs)
    return getter(*args, **kwargs)

  return custom_getter


def override_default_args(**kwargs):
  """Creates a custom getter that applies specified named arguments.

  The returned custom getter treats the specified named arguments as revised
  defaults, and does not override any non-`None` argument values supplied by
  the original get_variable call (or by a nested scope's custom getter).

  Args:
    **kwargs: Overriding arguments for the custom getter to use in preference
      the named arguments it's called with.

  Returns:
    Custom getter.
  """

  override_default_kwargs = kwargs

  def custom_getter(getter, *args, **kwargs):
    """Custom getter with certain named arguments overridden.

    Args:
      getter: Underlying variable getter to invoke.
      *args: Arguments, compatible with those of tf.get_variable.
      **kwargs: Keyword arguments, compatible with those of tf.get_variable.

    Returns:
      The result of invoking `getter(*args, **kwargs)` except that certain
      kwargs entries may have been overridden.
    """
    updated_kwargs = override_default_kwargs.copy()
    updated_kwargs.update({kw: value for kw, value in six.iteritems(kwargs)
                           if value is not None})
    return getter(*args, **updated_kwargs)

  return custom_getter
