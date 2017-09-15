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
"""Tests for sonnet.python.custom_getters.context."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf


def _suffix_getter(getter, name, *args, **kwargs):
  """Adds a suffix to the variable name (custom getter for use in tests)."""
  unused_original_variable = getter(name, *args, **kwargs)
  kwargs['reuse'] = None
  kwargs['trainable'] = False
  return getter(name + '_custom', *args, **kwargs)


class ContextTest(tf.test.TestCase):

  def testContextCallsCustomGetterOnlyWhenInScope(self):
    custom_getter = snt.custom_getters.Context(_suffix_getter)
    with tf.variable_scope('', custom_getter=custom_getter):
      lin = snt.Linear(10, name='linear')

    inputs = tf.placeholder(tf.float32, [10, 10])

    _ = lin(inputs)
    self.assertEqual('linear/w:0', lin.w.name)
    with custom_getter:
      _ = lin(inputs)
      self.assertEqual('linear/w_custom:0', lin.w.name)
    _ = lin(inputs)
    self.assertEqual('linear/w:0', lin.w.name)

  def testNestedContextCallsCustomGetterOnlyWhenInScope(self):
    custom_getter = snt.custom_getters.Context(_suffix_getter)
    with tf.variable_scope('', custom_getter=custom_getter):
      lin = snt.Linear(10, name='linear')

    inputs = tf.placeholder(tf.float32, [10, 10])
    with custom_getter:
      _ = lin(inputs)
      self.assertEqual('linear/w_custom:0', lin.w.name)
      with custom_getter:
        _ = lin(inputs)
        self.assertEqual('linear/w_custom:0', lin.w.name)
      _ = lin(inputs)
      self.assertEqual('linear/w_custom:0', lin.w.name)
    _ = lin(inputs)
    self.assertEqual('linear/w:0', lin.w.name)

  def testTwoContextsOperateIndependently(self):
    custom_getter1 = snt.custom_getters.Context(_suffix_getter)
    with tf.variable_scope('', custom_getter=custom_getter1):
      lin1 = snt.Linear(10, name='linear1')

    custom_getter2 = snt.custom_getters.Context(_suffix_getter)
    with tf.variable_scope('', custom_getter=custom_getter2):
      lin2 = snt.Linear(10, name='linear2')

    inputs = tf.placeholder(tf.float32, [10, 10])
    _ = lin1(inputs), lin2(inputs)
    with custom_getter1:
      _ = lin1(inputs), lin2(inputs)
      self.assertEqual('linear1/w_custom:0', lin1.w.name)
      self.assertEqual('linear2/w:0', lin2.w.name)
      with custom_getter2:
        _ = lin1(inputs), lin2(inputs)
        self.assertEqual('linear1/w_custom:0', lin1.w.name)
        self.assertEqual('linear2/w_custom:0', lin2.w.name)
      _ = lin1(inputs), lin2(inputs)
      self.assertEqual('linear1/w_custom:0', lin1.w.name)
      self.assertEqual('linear2/w:0', lin2.w.name)
    _ = lin1(inputs), lin2(inputs)
    self.assertEqual('linear1/w:0', lin1.w.name)
    self.assertEqual('linear2/w:0', lin2.w.name)


if __name__ == '__main__':
  tf.test.main()
