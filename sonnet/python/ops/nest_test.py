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
    self.assertEqual(flattened, ["z", 3, 4, 5, 1, 2, 3, 17])

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


