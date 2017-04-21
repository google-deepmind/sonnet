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


def _yield_flat_up_to(shallow_tree, input_tree):
  """Yields elements `input_tree` partially flattened up to `shallow_tree`."""
  if is_sequence(shallow_tree):
    for shallow_branch, input_branch in zip(shallow_tree, input_tree):
      for input_leaf in _yield_flat_up_to(shallow_branch, input_branch):
        yield input_leaf
  else:
    yield input_tree


def assert_shallow_structure(shallow_tree, input_tree):
  """Asserts that `shallow_tree` is a shallow structure of `input_tree`.

  That is, this function tests if the `input_tree` structure can be created from
  the `shallow_tree` structure by replacing its leaf nodes with deeper
  tree structures.

  Args:
    shallow_tree: an arbitrarily nested structure.
    input_tree: an arbitrarily nested structure.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  if is_sequence(shallow_tree):
    if not is_sequence(input_tree):
      raise TypeError(
          "If shallow structure is a sequence, input must also be a sequence. "
          "Input has type: %s." % type(input_tree))

    if not isinstance(input_tree, type(shallow_tree)):
      raise TypeError(
          "The two structures don't have the same sequence type. Input "
          "structure has type %s, while shallow structure has type %s."
          % (type(input_tree), type(shallow_tree)))

    if len(input_tree) != len(shallow_tree):
      raise ValueError(
          "The two structures don't have the same sequence length. Input "
          "structure has length %s, while shallow structure has length %s."
          % (len(input_tree), len(shallow_tree)))

    for shallow_branch, input_branch in zip(shallow_tree, input_tree):
      assert_shallow_structure(shallow_branch, input_branch)


def flatten_up_to(shallow_tree, input_tree):
  """Flattens `input_tree` up to `shallow_tree`.

  Any further depth in structure in `input_tree` is retained as elements in the
  partially flatten output.

  If `shallow_tree` and `input_tree` are not sequences, this returns a
  single-element list: `[input_tree]`.

  Use Case:

  Sometimes we may wish to partially flatten a nested sequence, retaining some
  of the nested structure. We achieve this by specifying a shallow structure,
  `shallow_tree`, we wish to flatten up to.

  The input, `input_tree`, can be thought of as having the same structure as
  `shallow_tree`, but with leaf nodes that are themselves tree structures.

  Examples:

  ```python
  input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
  shallow_tree = [[True, True], [False, True]]

  flattened_input_tree = flatten_up_to(shallow_tree, input_tree)
  flattened_shallow_tree = flatten_up_to(shallow_tree, shallow_tree)

  # Output is:
  # [[2, 2], [3, 3], [4, 9], [5, 5]]
  # [True, True, False, True]
  ```

  ```python
  input_tree = [[('a', 1), [('b', 2), [('c', 3), [('d', 4)]]]]]
  shallow_tree = [['level_1', ['level_2', ['level_3', ['level_4']]]]]

  input_tree_flattened_as_shallow_tree = flatten_up_to(shallow_tree, input_tree)
  input_tree_flattened = flatten(input_tree)

  # Output is:
  # [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
  # ['a', 1, 'b', 2, 'c', 3, 'd', 4]
  ```

  Non-Sequence Edge Cases:

  ```python
  flatten_up_to(0, 0)  # Output: [0]
  flatten_up_to(0, [0, 1, 2])  # Output: [[0, 1, 2]]
  flatten_up_to([0, 1, 2], 0)  # Output: TypeError
  flatten_up_to([0, 1, 2], [0, 1, 2])  # Output: [0, 1, 2]
  ```

  Args:
    shallow_tree: a possibly pruned structure of input_tree.
    input_tree: an arbitrarily nested structure or a scalar object.
      Note, numpy arrays are considered scalars.

  Returns:
    A Python list, the partially flattened version of `input_tree` according to
    the structure of `shallow_tree`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  assert_shallow_structure(shallow_tree, input_tree)
  return list(_yield_flat_up_to(shallow_tree, input_tree))


def map_up_to(shallow_tree, fn_or_op, *inputs):
  """Applies a function or op to a number of partially flattened inputs.

  The `inputs` are flattened up to `shallow_tree` before being mapped.

  Use Case:

  Sometimes we wish to apply a function to a partially flattened
  sequence (for example when the function itself takes sequence inputs). We
  achieve this by specifying a shallow structure, `shallow_tree` we wish to
  flatten up to.

  The `inputs`, can be thought of as having the same structure as
  `shallow_tree`, but with leaf nodes that are themselves tree structures.

  This function therefore will return something with the same base structure as
  `shallow_tree`.

  Examples:

  ```python
  ab_tuple = collections.namedtuple("ab_tuple", "a, b")
  op_tuple = collections.namedtuple("op_tuple", "add, mul")
  inp_val = ab_tuple(a=2, b=3)
  inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
  out = nest.map_up_to(inp_val, lambda val, ops: (val + ops.add) * ops.mul,
                       inp_val, inp_ops)

  # Output is: ab_tuple(a=6, b=15)
  ```

  ```python
  data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
  name_list = ['evens', ['odds', 'primes']]
  out = nest.map_up_to(name_list,
                      lambda name, sec: "first_{}_{}".format(len(sec), name),
                      name_list, data_list)

  # Output is: ['first_4_evens', ['first_5_odds', 'first_3_primes']]
  ```

  Args:
    shallow_tree: a shallow tree, common to all the inputs.
    fn_or_op: function or other callable which will be applied to each
        input individually.
    *inputs: arbitrarily nested combination of objects that are compatible with
        shallow_tree. The function fn_or_op is applied to corresponding
        partially flattened elements of each input, so the function must support
        arity of `len(inputs)`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.

  Returns:
    result of repeatedly applying `fn_or_op`, with same structure as
    `shallow_tree`.
  """
  if not inputs:
    raise ValueError("Cannot map over no sequences")
  for input_tree in inputs:
    assert_shallow_structure(shallow_tree, input_tree)

  # Flatten each input separately, apply the function to corresponding elements,
  # then repack based on the structure of the first input.
  all_flattened_up_to = [flatten_up_to(shallow_tree, input_tree)
                         for input_tree in inputs]
  results = [fn_or_op(*tensors) for tensors in zip(*all_flattened_up_to)]
  return nest.pack_sequence_as(structure=shallow_tree, flat_sequence=results)


def map(fn_or_op, *inputs):  # pylint: disable=redefined-builtin
  """Applies a function or op to a number of arbitrarily nested structures.

  The nested structure can be any combination of tuples and namedtuples. When
  calling `snt.nest.map(fun, data)` then `fun(d)` will be called for each
  leaf element inside `data`, and the results of these calls will be packaged
  into the same structure as `data` before being returned from map. If called
  with multiple input arguments, e.g. `snt.nest.map(fun, data_a, data_b, ...)`
  then the nesting structure of the `data_x` arguments must be identical. The
  output is the same nesting structure as the input, where each value is the
  result of calling `fun(d_a, d_b, ...)` for corresponding `d_x` values.

  ```python
  nt = collections.namedtuple('nt', 'a, b')
  inp_a = nt(a='foo', b=('bar', 'baz'))
  inp_b = nt(a=2, b=(1,3))
  out = snt.nest.map(lambda string, repeats: string * repeats, inp_a, inp_b)

  # Output is: nt(a='foofoo', b=('bar', 'bazbazbaz'))
  ```

  Note that this should always be accessed as `snt.nest.map` or `nest.map`, so
  we purposefully ignore the warning about redefining builtins.

  Args:
    fn_or_op: function or other callable which will be applied to each single
        input Tensor individually.
    *inputs: list of Tensors, or arbitrarily nested combination of Tensors. If
        more than one input is provided, these must have the same nesting
        structure. The function fn_or_op is applied to corresponding elements of
        each input, so the function must support arity of `len(inputs)`.

  Raises:
    ValueError: if any of the `inputs` do not share the same nesting structure.

  Returns:
    result of repeatedly applying `fn_or_op`, with same structure as `input[0]`.
  """
  if not inputs:
    raise ValueError("Cannot map over no sequences")
  for input_tree in inputs[1:]:
    assert_same_structure(inputs[0], input_tree)

  # Flatten each input separately, apply the function to corresponding elements,
  # then repack based on the structure of the first input.
  all_flattened = [flatten(input_tree) for input_tree in inputs]
  applied_results = [fn_or_op(*tensors) for tensors in zip(*all_flattened)]
  return nest.pack_sequence_as(structure=inputs[0],
                               flat_sequence=applied_results)


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
