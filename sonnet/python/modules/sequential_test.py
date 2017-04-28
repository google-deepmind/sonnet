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

"""Tests for sonnet.python.modules.nn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import six
import sonnet as snt
import tensorflow as tf


class SequentialTest(tf.test.TestCase):

  def testConstruct(self):
    module1 = snt.Linear(name="linear_1", output_size=123)
    module2 = snt.Linear(name="linear_2", output_size=45)

    seq = snt.Sequential([module1, tf.nn.relu, module2], name="sequential1")

    inputs = tf.placeholder(tf.float32, [67, 89])
    outputs = seq(inputs)

    self.assertEqual(outputs.get_shape().as_list(), [67, 45])

    self.assertEqual(seq.get_variables(), ())
    self.assertEqual(len(seq.layers), 3)

  def testConstructError(self):
    module1 = snt.Linear(name="linear_1", output_size=123)
    module2 = snt.Linear(name="linear_2", output_size=45)

    with self.assertRaisesRegexp(TypeError,
                                 "Items 1 not callable with types: int"):
      snt.Sequential([module1, 5, module2], name="sequential1")

    err_str = "Items 1, 2 not callable with types: int, bool"
    with self.assertRaisesRegexp(TypeError, err_str):
      snt.Sequential([module1, 5, True, module2], name="sequential1")

  def testTupleInput(self):
    def module1(a, b):
      return a, b

    _, _ = snt.Sequential([module1, module1], name="seq1")(1, 2)

    def module2(a, b, c):
      return a, b, c

    if six.PY3:
      err_str = r"module2\(\) missing 1 required positional argument: 'c'"
    else:
      err_str = r"module2\(\) takes exactly 3 arguments \(2 given\)"
    with self.assertRaisesRegexp(TypeError, err_str):
      _, _ = snt.Sequential([module1, module2], name="seq2")(1, 2)

  def testCopiesModules(self):
    modules = [snt.Linear(output_size=200), tf.tanh, snt.Linear(output_size=10)]
    sequential = snt.Sequential(modules)

    # Modify the list, to simulate PEBKAC. Sequential must make internal copy.
    modules[1] = "i'm a string, not a module"

    # Connecting the Sequential would produce a TypeError if `modules` was
    # stored by reference, rather than making a copy.
    sequential(tf.placeholder(tf.float32, [23, 42]))

  def testNoneFails(self):
    with self.assertRaisesRegexp(TypeError,
                                 "'NoneType' object is not iterable"):
      snt.Sequential(None)

  def testNameScopeRecording(self):
    lin = snt.Linear(output_size=256)
    sequential = snt.Sequential([lin])

    with tf.name_scope("blah"):
      sequential(tf.placeholder(dtype=tf.float32, shape=[2, 3]))
    self.assertEqual(sequential.name_scopes, ("blah/sequential",))
    self.assertEqual(lin.name_scopes, ("blah/sequential/linear",))


if __name__ == "__main__":
  tf.test.main()
