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

"""Utility functions for dealing with nested structures of Tensors.

These complement `nest.flatten` and `nest.pack_sequence_as` from the core TF
distribution.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

nest = tf.contrib.framework.nest

_DONE_WARN = {}


def with_deprecation_warning(fn, extra_message=''):
  """Wraps the function and prints a warn-once (per `extra_message`) warning."""
  def new_fn(*args, **kwargs):
    if extra_message not in _DONE_WARN:
      tf.logging.warning(
          'Sonnet nest is deprecated. Please use '
          'tf.contrib.framework.nest instead. '
          + extra_message
      )
      _DONE_WARN[extra_message] = True
    return fn(*args, **kwargs)
  return new_fn


assert_same_structure = with_deprecation_warning(nest.assert_same_structure)
flatten = with_deprecation_warning(nest.flatten)
flatten_iterable = with_deprecation_warning(
    nest.flatten,
    'In addition, `flatten_iterable` is renamed to `flatten`.'
)
is_sequence = with_deprecation_warning(nest.is_sequence)
is_iterable = with_deprecation_warning(
    nest.is_sequence,
    'In addition, `is_iterable` is renamed to `is_sequence`.'
)
pack_sequence_as = with_deprecation_warning(nest.pack_sequence_as)
map = with_deprecation_warning(  # pylint: disable=redefined-builtin
    nest.map_structure,
    'In addition, `map` is renamed to `map_structure`.'
)
map_up_to = with_deprecation_warning(
    nest.map_structure_up_to,
    'In addition, `map_up_to` is renamed to `map_structure_up_to`.'
)
assert_shallow_structure = with_deprecation_warning(
    nest.assert_shallow_structure)
flatten_up_to = with_deprecation_warning(nest.flatten_up_to)
flatten_dict_items = with_deprecation_warning(nest.flatten_dict_items)


def pack_iterable_as(structure, flat_iterable):
  """See `nest.pack_sequence_as`. Provided for named-arg compatibility."""
  return nest.pack_sequence_as(structure, flat_iterable)


pack_iterable_as = with_deprecation_warning(
    pack_iterable_as,
    'In addition, `pack_iterable_as` is renamed to `pack_sequence_as`.'
)
