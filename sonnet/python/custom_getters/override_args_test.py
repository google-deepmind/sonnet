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
"""Tests for sonnet.python.modules.custom_getters.stop_gradient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import sonnet as snt
import tensorflow as tf


def _suffix_custom_getter(getter, name, *args, **kwargs):
  return getter(name + "_test", *args, **kwargs)


class OverrideArgsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ("override_args", snt.custom_getters.override_args),
      ("override_default_args", snt.custom_getters.override_default_args),
  )
  def testUsage(self, custom_getter_fn):
    # Create a module with no custom getters.
    linear = snt.Linear(10)

    # Create a module within the scope of an 'override args' custom getter.
    local_custom_getter = custom_getter_fn(
        collections=[tf.GraphKeys.LOCAL_VARIABLES])
    with tf.variable_scope("", custom_getter=local_custom_getter):
      local_linear = snt.Linear(10)

    # Connect both modules to the graph, creating their variables.
    inputs = tf.placeholder(dtype=tf.float32, shape=(7, 11))
    linear(inputs)
    local_linear(inputs)

    self.assertIn(linear.w, tf.global_variables())
    self.assertNotIn(linear.w, tf.local_variables())
    self.assertIn(local_linear.w, tf.local_variables())
    self.assertNotIn(local_linear.w, tf.global_variables())

  @parameterized.named_parameters(
      ("override_args", snt.custom_getters.override_args),
      ("override_default_args", snt.custom_getters.override_default_args),
  )
  def testNestedWithin(self, custom_getter_fn):
    # Create a module with an 'override args' custom getter, within the scope
    # of another custom getter.
    local_custom_getter = custom_getter_fn(
        collections=[tf.GraphKeys.LOCAL_VARIABLES])
    with tf.variable_scope("", custom_getter=_suffix_custom_getter):
      local_linear = snt.Linear(10, custom_getter=local_custom_getter)

    # Connect the module to the graph, creating its variables.
    inputs = tf.placeholder(dtype=tf.float32, shape=(7, 11))
    local_linear(inputs)

    # Both custom getters should be effective.
    self.assertIn(local_linear.w, tf.local_variables())
    self.assertNotIn(local_linear.w, tf.global_variables())
    self.assertEqual("linear/w_test", local_linear.w.op.name)

  @parameterized.named_parameters(
      ("override_args", snt.custom_getters.override_args),
      ("override_default_args", snt.custom_getters.override_default_args),
  )
  def testWithNested(self, custom_getter_fn):
    # Create a module with a custom getter, within the scope of an
    # 'override args' custom getter.
    local_custom_getter = custom_getter_fn(
        collections=[tf.GraphKeys.LOCAL_VARIABLES])
    with tf.variable_scope("", custom_getter=local_custom_getter):
      local_linear = snt.Linear(10, custom_getter=_suffix_custom_getter)

    # Connect the module to the graph, creating its variables.
    inputs = tf.placeholder(dtype=tf.float32, shape=(7, 11))
    local_linear(inputs)

    # Both custom getters should be effective.
    self.assertIn(local_linear.w, tf.local_variables())
    self.assertNotIn(local_linear.w, tf.global_variables())
    self.assertEqual("linear/w_test", local_linear.w.op.name)

  def testExplicitArgOverridden(self):
    # Create a variable within the scope of an 'override args' custom getter.
    local_custom_getter = snt.custom_getters.override_args(
        collections=[tf.GraphKeys.LOCAL_VARIABLES])
    with tf.variable_scope("", custom_getter=local_custom_getter):
      # Explicitly specify an arg that disagrees with the custom getter.
      v = tf.get_variable("v", (), collections=[tf.GraphKeys.GLOBAL_VARIABLES])

    # The custom getter should win.
    self.assertIn(v, tf.local_variables())
    self.assertNotIn(v, tf.global_variables())

  def testExplicitArgNotOverridden(self):
    # Create a variable within an 'override default args' custom getter.
    local_custom_getter = snt.custom_getters.override_default_args(
        collections=[tf.GraphKeys.LOCAL_VARIABLES])
    with tf.variable_scope("", custom_getter=local_custom_getter):
      # Explicitly specify an arg that disagrees with the custom getter.
      v = tf.get_variable("v", (), collections=[tf.GraphKeys.GLOBAL_VARIABLES])

    # The custom getter should honour the explicitly specified arg.
    self.assertIn(v, tf.global_variables())
    self.assertNotIn(v, tf.local_variables())


if __name__ == "__main__":
  tf.test.main()
