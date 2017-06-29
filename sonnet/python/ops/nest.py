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

import collections

# Dependency imports
import numpy as np
import six


from tensorflow.python.util import nest


# Alias the nest functions from TF so users can just import this module rather
# than needing to import two separate ones.
assert_same_structure = nest.assert_same_structure
flatten = nest.flatten
is_sequence = nest.is_sequence
pack_sequence_as = nest.pack_sequence_as
map = nest.map_structure  # pylint: disable=redefined-builtin
map_up_to = nest.map_structure_up_to
assert_shallow_structure = nest.assert_shallow_structure
flatten_up_to = nest.flatten_up_to
flatten_dict_items = nest.flatten_dict_items


def _yield_flat_up_to(shallow_tree, input_tree):
  """Yields elements `input_tree` partially flattened up to `shallow_tree`."""
  if is_sequence(shallow_tree):
    for shallow_branch, input_branch in zip(shallow_tree, input_tree):
      for input_leaf in _yield_flat_up_to(shallow_branch, input_branch):
        yield input_leaf
  else:
    yield input_tree


def _sorted(dict_):
  """Returns a sorted list from the dict, with error if keys not sortable."""
  try:
    return sorted(six.iterkeys(dict_))
  except TypeError:
    raise TypeError("nest only supports dicts with sortable keys.")


def _iterable_like(instance, args):
  """Converts the iterable `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, `namedtuple`, `dict`, or
        `collections.NamedDict`.
    args: elements to be converted to a sequence.

  Returns:
    `args` with the type of `instance`.
  """
  if isinstance(instance, collections.OrderedDict):
    return collections.OrderedDict(zip(six.iterkeys(instance), args))
  elif isinstance(instance, dict):
    return dict(zip(_sorted(instance), args))
  elif (isinstance(instance, tuple) and
        hasattr(instance, "_fields") and
        isinstance(instance._fields, collections.Sequence) and
        all(isinstance(f, six.string_types) for f in instance._fields)):
    # This is a namedtuple
    return type(instance)(*args)
  else:
    # Not a namedtuple
    return type(instance)(args)


def _yield_value_from_iterable(iterable):
  if isinstance(iterable, dict):
    if isinstance(iterable, collections.OrderedDict):
      for key in iterable:
        yield iterable[key]
    else:
      # Iterate through dictionaries in a deterministic order.
      for key in _sorted(iterable):
        yield iterable[key]
  else:
    for value in iterable:
      yield value


def _yield_flat_nest_from_iterable(iterable):
  for n in _yield_value_from_iterable(iterable):
    if is_iterable(n):
      for ni in _yield_flat_nest_from_iterable(n):
        yield ni
    else:
      yield n


def is_iterable(seq):
  """Returns true if `seq` is iterable (apart from strings).

  Args:
    seq: an input sequence.

  Returns:
    True if `seq` is iterable, but not a string.
  """
  if isinstance(seq, six.string_types) or isinstance(seq, np.ndarray):
    return False
  try:
    iter(seq)
    return True
  except TypeError:
    return False


def flatten_iterable(structure):
  """Returns a flat sequence from a given nested structure.

  Unlike `flatten` (which just flattens sequences), this will also flatten
  dictionaries.

  If `nest` is not a sequence, this returns a single-element list: `[nest]`.

  Args:
    structure: an arbitrarily nested structure or a scalar object. Note, numpy
        arrays are considered scalars.

  Returns:
    A Python list, the flattened version of the input.

  Raises:
    TypeError: If `structure` is a dict or contains a dict with non-sortable
        keys.
  """
  if is_iterable(structure):
    return list(_yield_flat_nest_from_iterable(structure))
  else:
    return [structure]


def _packed_iterable_nest_with_indices(structure, flat, index):
  """Helper function for pack_nest_as.

  Args:
    structure: Substructure (tuple / dict /etc ) to mimic.
    flat: Flattened values to output substructure for.
    index: Index at which to start reading from flat.

  Returns:
    The tuple (new_index, child), where:
      * new_index - the updated index into `flat` having processed `structure`.
      * packed - the subset of `flat` corresponding to `structure`,
                 having started at `index`, and packed into the same nested
                 format.

  Raises:
    ValueError: if `structure` contains more elements than `flat`
      (assuming indexing starts from `index`).
  """
  packed = []
  for s in _yield_value_from_iterable(structure):
    if is_iterable(s):
      new_index, child = _packed_iterable_nest_with_indices(s, flat, index)
      packed.append(_iterable_like(s, child))
      index = new_index
    else:
      packed.append(flat[index])
      index += 1
  return index, packed


def pack_iterable_as(structure, flat_iterable):
  """Returns a given flattened iterable packed into a nest.

  If `structure` is a scalar, `flat_iterable` must be a single-element list;
  in this case the return value is `flat_sequence[0]`.

  Args:
    structure: non-string iterable (such as a tuple, list, dict, or named dict)
        constructed of scalars and/or other tuples/lists, or a scalar. Note:
        numpy arrays are considered scalars.
    flat_iterable: flat iterable to pack.

  Returns:
    packed: `flat_iterable` converted to have the same recursive structure as
      `structure`.

  Raises:
    ValueError: If nest and structure have different element counts.
    TypeError: If `structure` is a dict or contains a dict with non-sortable
        keys.
  """
  if not is_iterable(flat_iterable):
    raise TypeError("flat_iterable must be an iterable")

  if not is_iterable(structure):
    if len(flat_iterable) != 1:
      raise ValueError("Structure is a scalar but len(flat_iterable) == %d > 1"
                       % len(flat_iterable))
    return flat_iterable[0]

  flat_structure = flatten_iterable(structure)
  if len(flat_structure) != len(flat_iterable):
    raise ValueError(
        "Could not pack iterable. Structure had %d elements, but flat_iterable "
        "had %d elements.  Structure: %s, flat_iterable: %s."
        % (len(flat_structure), len(flat_iterable), structure, flat_iterable))

  _, packed = _packed_iterable_nest_with_indices(structure, flat_iterable, 0)
  return _iterable_like(structure, packed)
