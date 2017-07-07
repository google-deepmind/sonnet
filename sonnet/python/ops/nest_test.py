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

"""Tests for sonnet.python.ops.nest.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

import numpy as np
import six
from sonnet.python.ops import nest
import tensorflow as tf

typekw = "class" if six.PY3 else "type"


class NestTest(tf.test.TestCase):

  def testAssertShallowStructure(self):
    inp_ab = ["a", "b"]
    inp_abc = ["a", "b", "c"]
    with self.assertRaises(ValueError) as cm:
      nest.assert_shallow_structure(inp_abc, inp_ab)
    self.assertEqual(str(cm.exception),
                     "The two structures don't have the same sequence length. "
                     "Input structure has length 2, while shallow structure "
                     "has length 3.")

    inp_ab1 = [(1, 1), (2, 2)]
    inp_ab2 = [[1, 1], [2, 2]]
    with self.assertRaises(TypeError) as cm:
      nest.assert_shallow_structure(inp_ab2, inp_ab1)
    self.assertEqual(str(cm.exception),
                     "The two structures don't have the same sequence type. "
                     "Input structure has type <{0} 'tuple'>, while shallow "
                     "structure has type <{0} 'list'>.".format(typekw))

  def testFlattenUpTo(self):
    # Normal application (Example 1).
    input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
    shallow_tree = [[True, True], [False, True]]
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [[2, 2], [3, 3], [4, 9], [5, 5]])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])

    # Normal application (Example 2).
    input_tree = [[("a", 1), [("b", 2), [("c", 3), [("d", 4)]]]]]
    shallow_tree = [["level_1", ["level_2", ["level_3", ["level_4"]]]]]
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    input_tree_flattened = nest.flatten(input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    self.assertEqual(input_tree_flattened, ["a", 1, "b", 2, "c", 3, "d", 4])

    ## Shallow non-list edge-case.
    # Using iterable elements.
    input_tree = ["input_tree"]
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = ["input_tree_0", "input_tree_1"]
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = [0]
    shallow_tree = 9
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = [0, 1]
    shallow_tree = 9
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Both non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = 0
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Input non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = ["shallow_tree"]
    with self.assertRaises(TypeError) as cm:
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(str(cm.exception),
                     "If shallow structure is a sequence, input must also be "
                     "a sequence. Input has type: <{} 'str'>.".format(typekw))
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = "input_tree"
    shallow_tree = ["shallow_tree_9", "shallow_tree_8"]
    with self.assertRaises(TypeError) as cm:
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(str(cm.exception),
                     "If shallow structure is a sequence, input must also be "
                     "a sequence. Input has type: <{} 'str'>.".format(typekw))
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = [9]
    with self.assertRaises(TypeError) as cm:
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(str(cm.exception),
                     "If shallow structure is a sequence, input must also be "
                     "a sequence. Input has type: <{} 'int'>.".format(typekw))
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = 0
    shallow_tree = [9, 8]
    with self.assertRaises(TypeError) as cm:
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(str(cm.exception),
                     "If shallow structure is a sequence, input must also be "
                     "a sequence. Input has type: <{} 'int'>.".format(typekw))
    self.assertEqual(flattened_shallow_tree, shallow_tree)

  def testMapUpTo(self):
    # Example 1.
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    op_tuple = collections.namedtuple("op_tuple", "add, mul")
    inp_val = ab_tuple(a=2, b=3)
    inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
    out = nest.map_up_to(inp_val, lambda val, ops: (val + ops.add) * ops.mul,
                         inp_val, inp_ops)
    self.assertEqual(out.a, 6)
    self.assertEqual(out.b, 15)

    # Example 2.
    data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
    name_list = ["evens", ["odds", "primes"]]
    out = nest.map_up_to(name_list,
                         lambda name, sec: "first_{}_{}".format(len(sec), name),
                         name_list, data_list)
    self.assertEqual(out, ["first_4_evens", ["first_5_odds", "first_3_primes"]])

  def testStringRepeat(self):
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    inp_a = ab_tuple(a="foo", b=("bar", "baz"))
    inp_b = ab_tuple(a=2, b=(1, 3))
    out = nest.map(lambda string, repeats: string * repeats, inp_a, inp_b)
    self.assertEqual(out.a, "foofoo")
    self.assertEqual(out.b[0], "bar")
    self.assertEqual(out.b[1], "bazbazbaz")

  def testMapSingleCollection(self):
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    nt = ab_tuple(a=("something", "something_else"),
                  b="yet another thing")
    rev_nt = nest.map(lambda x: x[::-1], nt)

    # Check the output is the correct structure, and all strings are reversed.
    nest.assert_same_structure(nt, rev_nt)
    self.assertEqual(nt.a[0][::-1], rev_nt.a[0])
    self.assertEqual(nt.a[1][::-1], rev_nt.a[1])
    self.assertEqual(nt.b[::-1], rev_nt.b)

  def testMapOverTwoTuples(self):
    inp_a = (tf.placeholder(tf.float32, shape=[3, 4]),
             tf.placeholder(tf.float32, shape=[3, 7]))
    inp_b = (tf.placeholder(tf.float32, shape=[3, 4]),
             tf.placeholder(tf.float32, shape=[3, 7]))

    output = nest.map(lambda x1, x2: x1 + x2, inp_a, inp_b)

    nest.assert_same_structure(output, inp_a)
    self.assertShapeEqual(np.zeros((3, 4)), output[0])
    self.assertShapeEqual(np.zeros((3, 7)), output[1])

    feed_dict = {
        inp_a: (np.random.randn(3, 4), np.random.randn(3, 7)),
        inp_b: (np.random.randn(3, 4), np.random.randn(3, 7))
    }

    with self.test_session() as sess:
      output_np = sess.run(output, feed_dict=feed_dict)
    self.assertAllClose(output_np[0],
                        feed_dict[inp_a][0] + feed_dict[inp_b][0])
    self.assertAllClose(output_np[1],
                        feed_dict[inp_a][1] + feed_dict[inp_b][1])

  def testStructureMustBeSame(self):
    inp_a = (3, 4)
    inp_b = (42, 42, 44)
    err = "The two structures don't have the same number of elements."
    with self.assertRaisesRegexp(ValueError, err):
      nest.map(lambda a, b: a + b, inp_a, inp_b)

  def testMultiNest(self):
    inp_a = (3, (4, 5))
    inp_b = (42, (42, 44))
    output = nest.map(lambda a, b: a + b, inp_a, inp_b)
    self.assertEqual((45, (46, 49)), output)

  def testNoSequences(self):
    with self.assertRaisesRegexp(ValueError,
                                 "Must provide at least one structure"):
      nest.map(lambda x: x)

  def testEmptySequences(self):
    f = lambda x: x + 1
    empty_nt = collections.namedtuple("empty_nt", "")

    self.assertEqual((), nest.map(f, ()))
    self.assertEqual([], nest.map(f, []))
    self.assertEqual(empty_nt(), nest.map(f, empty_nt()))

    # This is checking actual equality of types, empty list != empty tuple
    self.assertNotEqual((), nest.map(f, []))

  def testFlattenAndPackIterable(self):
    # A nice messy mix of tuples, lists, dicts, and `OrderedDict`s.
    named_tuple = collections.namedtuple("A", ("b", "c"))
    mess = [
        "z",
        named_tuple(3, 4),
        {
            "c": [
                1,
                collections.OrderedDict([
                    ("b", 3),
                    ("a", 2),
                ]),
            ],
            "b": 5
        },
        17
    ]

    flattened = nest.flatten_iterable(mess)
    self.assertEqual(flattened, ["z", 3, 4, 5, 1, 3, 2, 17])

    structure_of_mess = [
        14,
        named_tuple("a", True),
        {
            "c": [
                0,
                collections.OrderedDict([
                    ("b", 9),
                    ("a", 8),
                ]),
            ],
            "b": 3
        },
        "hi everybody",
    ]

    unflattened = nest.pack_iterable_as(structure_of_mess, flattened)
    self.assertEqual(unflattened, mess)

  def testFlattenIterable_numpyIsNotFlattened(self):
    structure = np.array([1, 2, 3])
    flattened = nest.flatten_iterable(structure)
    self.assertEqual(len(flattened), 1)

  def testFlattenIterable_stringIsNotFlattened(self):
    structure = "lots of letters"
    flattened = nest.flatten_iterable(structure)
    self.assertEqual(len(flattened), 1)

  def testFlatternIterable_scalarStructure(self):
    # Tests can call flatten_iterable with single "scalar" object.
    structure = "hello"
    flattened = nest.flatten_iterable(structure)
    unflattened = nest.pack_iterable_as("goodbye", flattened)
    self.assertEqual(structure, unflattened)

  def testPackIterableAs_notIterableError(self):
    with self.assertRaisesRegexp(TypeError,
                                 "flat_iterable must be an iterable"):
      nest.pack_iterable_as("hi", "bye")

  def testPackIterableAs_scalarStructureError(self):
    with self.assertRaisesRegexp(
        ValueError, r"Structure is a scalar but len\(flat_iterable\) == 2 > 1"):
      nest.pack_iterable_as("hi", ["bye", "twice"])

  def testPackIterableAs_wrongLengthsError(self):
    with self.assertRaisesRegexp(
        ValueError,
        "Structure had 2 elements, but flat_iterable had 3 elements."):
      nest.pack_iterable_as(["hello", "world"],
                            ["and", "goodbye", "again"])


if __name__ == "__main__":
  tf.test.main()


