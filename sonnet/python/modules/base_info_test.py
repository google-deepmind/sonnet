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

"""Tests for sonnet.python.modules.base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
from sonnet.python.modules import base
from sonnet.python.modules import base_info
from sonnet.python.modules import basic
import tensorflow as tf
from tensorflow.python.util import nest

logging = tf.logging

THIS_MODULE = "__main__"
LINEAR_MODULE = "sonnet.python.modules.basic"

DumbNamedTuple = collections.namedtuple("DumbNamedTuple", ("arg1", "arg2"))


class NotATensor(object):
  pass


class DumbModule(base.AbstractModule):
  """Dumb module to test ModuleInfo."""

  def __init__(self, name, no_nest=False):
    base.AbstractModule.__init__(self, name=name)
    self.no_nest = no_nest

  def _build(self, inputs):
    if isinstance(inputs, (NotATensor, tf.SparseTensor)):
      outputs = inputs
    else:
      if self.no_nest:
        outputs = inputs
      else:
        outputs = nest.map_structure(tf.identity, inputs)
    return outputs


def _copy_default_graph():
  # Save default graph into `meta_graph_def`.
  meta_graph_def = tf.train.export_meta_graph()
  # Reset default graph.
  tf.reset_default_graph()
  # Load default graph from `meta_graph_def`.
  tf.train.import_meta_graph(meta_graph_def)


class ModuleInfoTest(tf.test.TestCase):

  def testIsNamedTuple(self):
    self.assertTrue(base_info._is_namedtuple(DumbNamedTuple(1, 2)))
    self.assertFalse(base_info._is_namedtuple((1, 2, 3)))
    self.assertFalse(base_info._is_namedtuple([1, 2, 3]))
    self.assertFalse(base_info._is_namedtuple(NotATensor()))

  def testIsIterable(self):
    self.assertTrue(base_info._is_iterable((1, 2, 3)))
    self.assertTrue(base_info._is_iterable([1, 2, 3]))
    self.assertTrue(base_info._is_iterable({1: 1, 2: 2, 3: 3}))
    self.assertTrue(base_info._is_iterable(
        collections.OrderedDict([(1, 1), (2, 2)])))
    self.assertTrue(base_info._is_iterable(DumbNamedTuple(1, 2)))
    tensor = tf.placeholder(dtype=tf.float32, shape=(1, 10,))
    self.assertFalse(base_info._is_iterable(set([1, 2, 3])))
    self.assertFalse(base_info._is_iterable(tensor))
    sparse_tensor = tf.SparseTensor(
        indices=tf.placeholder(dtype=tf.int64, shape=(10, 2,)),
        values=tf.placeholder(dtype=tf.float32, shape=(10,)),
        dense_shape=tf.placeholder(dtype=tf.int64, shape=(2,)))
    self.assertFalse(base_info._is_iterable(sparse_tensor))
    self.assertFalse(base_info._is_iterable(NotATensor()))
    self.assertFalse(base_info._is_iterable("foo"))
    def generator():
      for count in xrange(3):
        self.assertFalse(False)
        yield count
    self.assertFalse(base_info._is_iterable(generator))

  def testModuleInfo_multiple_modules(self):
    # pylint: disable=not-callable
    tf.reset_default_graph()
    dumb = DumbModule(name="dumb")
    dumb_1 = DumbModule(name="dumb")
    linear = basic.Linear(10, name="linear")
    ph_0 = tf.placeholder(dtype=tf.float32, shape=(1, 10,))
    dumb(ph_0)
    with tf.name_scope("foo"):
      dumb_1(ph_0)
    linear(ph_0)
    def check():
      sonnet_collection = tf.get_default_graph().get_collection(
          base_info.SONNET_COLLECTION_NAME)
      self.assertEqual(len(sonnet_collection), 3)
      # item 0.
      self.assertEqual(sonnet_collection[0].module_name, "dumb")
      self.assertEqual(sonnet_collection[0].class_name,
                       "{}.DumbModule".format(THIS_MODULE))
      self.assertEqual(sonnet_collection[0].scope_name, "dumb")
      self.assertEqual(len(sonnet_collection[0].connected_subgraphs), 1)
      self.assertEqual(
          sonnet_collection[0].connected_subgraphs[0].name_scope, "dumb")
      # item 1.
      self.assertEqual(sonnet_collection[1].module_name, "dumb_1")
      self.assertEqual(sonnet_collection[1].scope_name, "dumb_1")
      self.assertEqual(sonnet_collection[1].class_name,
                       "{}.DumbModule".format(THIS_MODULE))
      self.assertEqual(sonnet_collection[1].scope_name, "dumb_1")
      self.assertEqual(len(sonnet_collection[1].connected_subgraphs), 1)
      self.assertEqual(
          sonnet_collection[1].connected_subgraphs[0].name_scope, "foo/dumb_1")
      # item 2.
      self.assertEqual(sonnet_collection[2].module_name, "linear")
      self.assertEqual(sonnet_collection[2].scope_name, "linear")
      self.assertEqual(sonnet_collection[2].class_name,
                       "{}.Linear".format(LINEAR_MODULE))
      self.assertEqual(sonnet_collection[2].scope_name, "linear")
      self.assertEqual(len(sonnet_collection[2].connected_subgraphs), 1)
      self.assertEqual(
          sonnet_collection[2].connected_subgraphs[0].name_scope, "linear")
    check()
    _copy_default_graph()
    check()

  def testModuleInfo_multiple_subgraph(self):
    # pylint: disable=not-callable
    tf.reset_default_graph()
    dumb = DumbModule(name="dumb_a")
    ph_0 = tf.placeholder(dtype=tf.float32, shape=(1, 10,))
    dumb(ph_0)
    with tf.name_scope("foo"):
      dumb(ph_0)
    def check():
      sonnet_collection = tf.get_default_graph().get_collection(
          base_info.SONNET_COLLECTION_NAME)
      self.assertEqual(len(sonnet_collection), 1)
      self.assertEqual(len(sonnet_collection[0].connected_subgraphs), 2)
      connected_subgraph_0 = sonnet_collection[0].connected_subgraphs[0]
      connected_subgraph_1 = sonnet_collection[0].connected_subgraphs[1]
      self.assertEqual(connected_subgraph_0.name_scope, "dumb_a")
      self.assertEqual(connected_subgraph_1.name_scope, "foo/dumb_a")
    check()
    _copy_default_graph()
    check()

  def testModuleInfo_tensor(self):
    # pylint: disable=not-callable
    tf.reset_default_graph()
    dumb = DumbModule(name="dumb_a")
    ph_0 = tf.placeholder(dtype=tf.float32, shape=(1, 10,))
    dumb(ph_0)
    def check():
      sonnet_collection = tf.get_default_graph().get_collection(
          base_info.SONNET_COLLECTION_NAME)
      connected_subgraph = sonnet_collection[0].connected_subgraphs[0]
      self.assertIsInstance(connected_subgraph.inputs["inputs"], tf.Tensor)
      self.assertIsInstance(connected_subgraph.outputs, tf.Tensor)
    check()
    _copy_default_graph()
    check()

  def testModuleInfo_sparsetensor(self):
    # pylint: disable=not-callable
    tf.reset_default_graph()
    dumb = DumbModule(name="dumb_a")
    sparse_tensor = tf.SparseTensor(
        indices=tf.placeholder(dtype=tf.int64, shape=(10, 2,)),
        values=tf.placeholder(dtype=tf.float32, shape=(10,)),
        dense_shape=tf.placeholder(dtype=tf.int64, shape=(2,)))
    dumb(sparse_tensor)
    def check():
      sonnet_collection = tf.get_default_graph().get_collection(
          base_info.SONNET_COLLECTION_NAME)
      connected_subgraph = sonnet_collection[0].connected_subgraphs[0]
      self.assertIsInstance(
          connected_subgraph.inputs["inputs"], tf.SparseTensor)
      self.assertIsInstance(connected_subgraph.outputs, tf.SparseTensor)
    check()
    _copy_default_graph()
    check()

  def testModuleInfo_tuple(self):
    # pylint: disable=not-callable
    tf.reset_default_graph()
    dumb = DumbModule(name="dumb_a")
    ph_0 = tf.placeholder(dtype=tf.float32, shape=(1, 10,))
    ph_1 = tf.placeholder(dtype=tf.float32, shape=(1, 10,))
    dumb((ph_0, ph_1))
    def check():
      sonnet_collection = tf.get_default_graph().get_collection(
          base_info.SONNET_COLLECTION_NAME)
      connected_subgraph = sonnet_collection[0].connected_subgraphs[0]
      self.assertIsInstance(connected_subgraph.inputs["inputs"], tuple)
      self.assertIsInstance(connected_subgraph.outputs, tuple)
    check()
    _copy_default_graph()
    check()

  def testModuleInfo_namedtuple(self):
    # pylint: disable=not-callable
    tf.reset_default_graph()
    dumb = DumbModule(name="dumb_a")
    ph_0 = tf.placeholder(dtype=tf.float32, shape=(1, 10,))
    ph_1 = tf.placeholder(dtype=tf.float32, shape=(1, 10,))
    dumb(DumbNamedTuple(ph_0, ph_1))
    def check():
      sonnet_collection = tf.get_default_graph().get_collection(
          base_info.SONNET_COLLECTION_NAME)
      connected_subgraph = sonnet_collection[0].connected_subgraphs[0]
      self.assertTrue(
          base_info._is_namedtuple(connected_subgraph.inputs["inputs"]))
      self.assertTrue(base_info._is_namedtuple(connected_subgraph.outputs))
    check()
    _copy_default_graph()
    check()

  def testModuleInfo_dict(self):
    # pylint: disable=not-callable
    tf.reset_default_graph()
    dumb = DumbModule(name="dumb_a")
    ph_0 = tf.placeholder(dtype=tf.float32, shape=(1, 10,))
    ph_1 = tf.placeholder(dtype=tf.float32, shape=(1, 10,))
    dumb({"ph_0": ph_0, "ph_1": ph_1})
    def check():
      sonnet_collection = tf.get_default_graph().get_collection(
          base_info.SONNET_COLLECTION_NAME)
      connected_subgraph = sonnet_collection[0].connected_subgraphs[0]
      self.assertIsInstance(connected_subgraph.inputs["inputs"], dict)
      self.assertIsInstance(connected_subgraph.outputs, dict)
    check()
    _copy_default_graph()
    check()

  def testModuleInfo_not_a_tensor(self):
    # pylint: disable=not-callable
    tf.reset_default_graph()
    dumb = DumbModule(name="dumb_a")
    dumb(NotATensor())
    def check(check_type):
      sonnet_collection = tf.get_default_graph().get_collection(
          base_info.SONNET_COLLECTION_NAME)
      connected_subgraph = sonnet_collection[0].connected_subgraphs[0]
      self.assertIsInstance(connected_subgraph.inputs["inputs"], check_type)
      self.assertIsInstance(connected_subgraph.outputs, check_type)
    check(NotATensor)
    _copy_default_graph()
    check(base_info._UnserializableObject)

  def testModuleInfo_recursion(self):
    # pylint: disable=not-callable
    tf.reset_default_graph()
    dumb = DumbModule(name="dumb_a", no_nest=True)
    ph_0 = tf.placeholder(dtype=tf.float32, shape=(1, 10,))
    val = {"one": ph_0, "self": None}
    val["self"] = val
    dumb(val)
    def check(check_type):
      sonnet_collection = tf.get_default_graph().get_collection(
          base_info.SONNET_COLLECTION_NAME)
      connected_subgraph = sonnet_collection[0].connected_subgraphs[0]
      self.assertIsInstance(connected_subgraph.inputs["inputs"]["one"],
                            tf.Tensor)
      self.assertIsInstance(
          connected_subgraph.inputs["inputs"]["self"], check_type)
      self.assertIsInstance(connected_subgraph.outputs["one"], tf.Tensor)
      self.assertIsInstance(connected_subgraph.outputs["self"], check_type)
    check(dict)
    _copy_default_graph()
    check(base_info._UnserializableObject)


if __name__ == "__main__":
  tf.test.main()
