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
# =============================================================================
"""Tests for sonnet.experimental."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import sonnet as snt
import tensorflow as tf


class ReuseVarsTest(tf.test.TestCase):

  class VariableContainer(object):

    def __init__(self, name):
      with tf.variable_scope(name) as vs:
        self.variable_scope = vs

    @snt.experimental.reuse_vars
    def method_with_reuse(self):
      return tf.get_variable("a", shape=[1])

    def method_without_reuse(self):
      return tf.get_variable("b", shape=[1])

  class InheritedVariableContainer(VariableContainer):

    @snt.experimental.reuse_vars
    def not_inherited_method_with_reuse(self):
      return tf.get_variable("c", shape=[1])

  def test_reuse_method(self):
    obj1 = ReuseVarsTest.VariableContainer("scope1")
    obj2 = ReuseVarsTest.VariableContainer("scope2")

    self.assertEqual("b:0", obj1.method_without_reuse().name)
    self.assertRaisesRegexp(ValueError,
                            r"Variable b already exists, disallowed.*",
                            obj1.method_without_reuse)
    self.assertRaisesRegexp(ValueError,
                            r"Variable b already exists, disallowed.*",
                            obj2.method_without_reuse)

    self.assertEqual("scope1/a:0",
                     obj1.method_with_reuse().name)
    self.assertEqual("scope1/a:0",
                     obj1.method_with_reuse().name)

    self.assertEqual("scope2/a:0",
                     obj2.method_with_reuse().name)
    self.assertEqual("scope2/a:0",
                     obj2.method_with_reuse().name)

  def test_reuse_inherited_method(self):
    obj1 = ReuseVarsTest.InheritedVariableContainer("scope1")
    obj2 = ReuseVarsTest.InheritedVariableContainer("scope2")

    self.assertEqual("b:0", obj1.method_without_reuse().name)
    self.assertRaisesRegexp(ValueError,
                            r"Variable b already exists, disallowed.*",
                            obj1.method_without_reuse)
    self.assertRaisesRegexp(ValueError,
                            r"Variable b already exists, disallowed.*",
                            obj2.method_without_reuse)

    self.assertEqual("scope1/a:0", obj1.method_with_reuse().name)
    self.assertEqual("scope1/a:0", obj1.method_with_reuse().name)
    self.assertEqual("scope1/c:0", obj1.not_inherited_method_with_reuse().name)
    self.assertEqual("scope1/c:0", obj1.not_inherited_method_with_reuse().name)

    self.assertEqual("scope2/a:0", obj2.method_with_reuse().name)
    self.assertEqual("scope2/a:0", obj2.method_with_reuse().name)
    self.assertEqual("scope2/c:0", obj2.not_inherited_method_with_reuse().name)
    self.assertEqual("scope2/c:0", obj2.not_inherited_method_with_reuse().name)

  def test_reuse_abstract_module(self):

    class ModuleReuse(snt.AbstractModule):

      def __init__(self, shape, name="multi_template_test"):
        super(ModuleReuse, self).__init__(name)
        self._shape = shape

      @snt.experimental.reuse_vars
      def a(self):
        return tf.get_variable("a", shape=self._shape)

      @snt.experimental.reuse_vars
      def add_b(self, inputs):
        return inputs + tf.get_variable("b", shape=self._shape)

      def _build(self, inputs):
        return self.add_b(inputs + self.a())

    np.random.seed(100)
    batch_size = 3
    in_size = 4
    inputs = tf.placeholder(tf.float32, shape=[batch_size, in_size])

    module1 = ModuleReuse(inputs.get_shape().as_list())
    module2 = ModuleReuse(inputs.get_shape().as_list())

    a1 = module1.a()
    inputs_plus_b1 = module1.add_b(inputs)
    inputs_plus_ab1 = module1(inputs)  # pylint: disable=not-callable

    inputs_plus_ab2 = module2(inputs)  # pylint: disable=not-callable
    inputs_plus_b2 = module2.add_b(inputs)
    a2 = module2.a()

    inputs_plus_ab1_again = module1(inputs)  # pylint: disable=not-callable
    inputs_plus_ab2_again = module2(inputs)  # pylint: disable=not-callable

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      input_data = np.random.rand(batch_size, in_size)
      out = sess.run([a1, inputs_plus_b1, inputs_plus_ab1, a2, inputs_plus_b2,
                      inputs_plus_ab2],
                     feed_dict={inputs: input_data})

      self.assertNotAlmostEqual(np.linalg.norm(out[0] - out[3]), 0)
      self.assertNotAlmostEqual(np.linalg.norm(out[1] - out[4]), 0)
      self.assertNotAlmostEqual(np.linalg.norm(out[2] - out[5]), 0)

      self.assertAllClose(out[0] + out[1], out[2])
      self.assertAllClose(out[3] + out[4], out[5])

      out = sess.run([inputs_plus_ab1, inputs_plus_ab1_again],
                     feed_dict={inputs: input_data})
      self.assertAllEqual(out[0], out[1])

      out = sess.run([inputs_plus_ab2, inputs_plus_ab2_again],
                     feed_dict={inputs: input_data})
      self.assertAllEqual(out[0], out[1])

  def test_variable_scope_call_order(self):
    class TestModule(snt.AbstractModule):

      def __init__(self, name="test_module"):
        super(TestModule, self).__init__(name)

      @snt.experimental.reuse_vars
      def a(self):
        return self.scope_name

      def _build(self):
        pass

      @property
      def variable_scope(self):
        # Needed to access `self.variable_scope` before calling `self.build()`.
        return self._template.variable_scope

    m1 = TestModule(name="m1")
    m2 = TestModule(name="m2")

    a1 = m1.a
    a2 = m2.a

    print("VAR SCOPE:", m1.variable_scope)
    self.assertEqual("m1", a1())
    self.assertEqual("m2", a2())


if __name__ == "__main__":
  tf.test.main()
